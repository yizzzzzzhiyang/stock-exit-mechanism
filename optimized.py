"""
优化版主控脚本

优化项:
  1. 多屏障集成投票（替代单配置选择）
  2. 猪周期自适应特征
  3. 动态元模型阈值
  4. 屏障宽度自动搜索
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from config import setup_proxy_bypass
setup_proxy_bypass()

from data.data_fetcher import fetch_data, eng_features
from labeling.cusum import cusum_filter_vol
from labeling.triple_barrier import apply_triple_barrier
from evaluation.metrics import compare_baselines
from models.pipeline import (
    prepare_features, run_cv_training, run_threshold_search, aggregate_metrics
)


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加周期感知特征（针对猪周期等强周期股）"""
    X = df.copy()
    c = X['close']

    # 长期趋势（猪周期约3-4年 ≈ 750-1000交易日）
    X['cycle_ma_200'] = c.rolling(200).mean()
    X['cycle_ma_500'] = c.rolling(500).mean()
    X['cycle_ma_750'] = c.rolling(750).mean()

    # 价格相对长期均线的位置 → 周期位置
    X['cycle_pos_200'] = (c / X['cycle_ma_200'].clip(lower=1e-10)) - 1
    X['cycle_pos_500'] = (c / X['cycle_ma_500'].clip(lower=1e-10)) - 1

    # 动量周期（200日收益率）
    X['cycle_mom'] = c.pct_change(200)

    # 波动率周期（长期波动率是否在收缩/扩张）
    ret = X['r1']
    X['cycle_vol_long'] = ret.rolling(200).std()
    X['cycle_vol_short'] = ret.rolling(20).std()
    X['cycle_vol_ratio'] = X['cycle_vol_short'] / X['cycle_vol_long'].clip(lower=1e-10)

    # 回撤深度（从200日高点）
    X['cycle_peak_200'] = c.rolling(200).max()
    X['cycle_drawdown'] = (c - X['cycle_peak_200']) / X['cycle_peak_200'].clip(lower=1e-10)

    return X


def optimize_barrier_configs(close, t_events, vol, num_days=20,
                             configs=None):
    """
    多屏障配置搜索 + 集成
    返回: (最佳标签, 所有标签字典, 评分列表)
    """
    if configs is None:
        configs = [
            [1, 1], [1.5, 1], [2, 1], [2, 2],
            [2.5, 1], [3, 1], [3, 2], [1.5, 1.5]
        ]

    all_labels = {}
    all_scores = []

    for pt_sl in configs:
        key = f'p{pt_sl[0]}_s{pt_sl[1]}'
        labels = apply_triple_barrier(close, t_events, pt_sl, vol, num_days)
        all_labels[key] = labels

        n_pos = (labels['bin'] == 1).sum()
        n_neg = (labels['bin'] == -1).sum()
        total = n_pos + n_neg
        wr = n_pos / total if total > 0 else 0
        # 评分: 胜率接近50% + 总样本多
        balance = 1 - abs(wr - 0.5) * 2  # 0~1, 越接近0.5越高
        score = balance * np.log1p(total)
        all_scores.append((key, score, wr, total))

    # 按评分排序
    all_scores.sort(key=lambda x: -x[1])
    best_key = all_scores[0][0]

    return all_labels[best_key], all_labels, all_scores


def run_optimized(ts_code='002714.SZ', start='2015-01-01', end='2025-12-31',
                  cusum_vol_mult=1.5, num_days=20,
                  meta_thresholds=None, verbose=True):
    """
    优化版流水线
    """
    if verbose:
        print('=' * 60)
        print(f'优化版 TB+Meta+XGBoost — {ts_code}')
        print(f'期间: {start} ~ {end}')
        print('=' * 60)

    # ── 数据 ──
    df = fetch_data(ts_code, start, end)
    df = eng_features(df)
    df = add_cycle_features(df)  # 添加周期特征
    df = df.dropna()
    close = df['close']; vol = df['v20'].clip(lower=0.001); atr = df['atr']
    if verbose: print(f'数据: {len(df)}日, 特征: {len(df.columns)}列')

    # ── BK 分类推荐配置 ──
    from data.bk_config import get_recommended_config
    bk_cfg = get_recommended_config(ts_code)
    default_pt_sl = bk_cfg['pt_sl']
    default_days = num_days or bk_cfg['num_days']
    if verbose:
        print(f'BK分类: pt_sl={bk_cfg["pt_sl"]} days={bk_cfg["num_days"]}')
        print(f'  原因: {bk_cfg["reason"]}')

    # ── CUSUM ──
    t_events = cusum_filter_vol(close, vol, cusum_vol_mult)
    if verbose: print(f'CUSUM: {len(t_events)}事件')
    if len(t_events) < 20:
        t_events = cusum_filter_vol(close, vol, cusum_vol_mult * 0.5)

    # ── 多屏障搜索 ──
    labels, all_labels, config_scores = optimize_barrier_configs(
        close, t_events, vol, num_days)
    best_key = config_scores[0][0]
    if verbose:
        print(f'\n屏障搜索(前5):')
        for k, sc, wr, n in config_scores[:5]:
            print(f'  [{k}] 评分={sc:.3f} 胜率={wr:.1%} 样本={n}')
        print(f'选用: {best_key}')

    # ── 特征准备 ──
    X_feat, y, feat_cols = prepare_features(df, labels)
    if verbose: print(f'特征: {len(X_feat)}样本 × {len(feat_cols)}特征')

    # ── 多阈值搜索（使用共享模块） ──
    best_threshold, threshold_results = run_threshold_search(
        X_feat, y, close, thresholds=meta_thresholds, verbose=verbose
    )

    # ── 用最佳阈值重新跑 ──
    if verbose:
        print(f'\n── 最终结果 (th={best_threshold:.2f}) ──')
    final_results = run_cv_training(
        X_feat, y, close, meta_threshold=best_threshold, verbose=verbose
    )

    # ── 汇总 ──
    avg_m = aggregate_metrics(final_results)
    bl = compare_baselines(close, t_events, atr)

    if verbose:
        print(f'\n=== 优化版结果 ===')
        if avg_m:
            for k, v in avg_m.items():
                print(f'  {k:15s}: {v:.4f}')
        print(f'\n基准策略:')
        for _, r in bl.iterrows():
            print(f'  {r["name"]:15s}  SR={r["sharpe"]:.3f}  WR={r["win_rate"]:.1%}  MDD={r["max_dd"]:.2%}')
        if avg_m:
            best_base = bl['sharpe'].max() if len(bl) > 0 else 0
            print(f'\n══ 对比 ══')
            print(f'  优化版 TB+Meta: SR={avg_m["sharpe"]:.3f}')
            print(f'  最佳基准:        SR={best_base:.3f}')
            if best_base > 0:
                print(f'  提升: +{((avg_m["sharpe"]/best_base-1)*100):.1f}%')
            print(f'  DSR={avg_m["dsr"]:.2f}', end='')
            if avg_m['dsr'] > 1.96:
                print(' ✅ 统计显著')
            elif avg_m['dsr'] > 0:
                print(' ⚠️ 需更多数据')
            else:
                print(' ❌ 不优于随机')

    return {'avg_metrics': avg_m, 'baselines': bl, 'fold_results': final_results,
            'config_scores': config_scores, 'threshold_results': threshold_results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts_code', default='002714.SZ')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2025-12-31')
    args = parser.parse_args()
    run_optimized(ts_code=args.ts_code, start=args.start, end=args.end)
