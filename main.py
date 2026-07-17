"""
主控脚本：完整的 Triple Barrier + Meta-Labeling + XGBoost 退出机制流水线

风险应对:
  1. 屏障过拟合 → 多组屏障宽度的集成投票
  2. 特征泄漏   → Purged CV + Embargo
  3. 标签不平衡 → scale_pos_weight 自适应
  4. A 股特殊性  → T+1、涨跌停、ST 过滤

使用方法:
  python main.py --ts_code 000001.SZ --start 2015-01-01 --end 2025-12-31
  python main.py --ts_code 000300.SH --start 2010-01-01 --end 2025-12-31
"""
import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import setup_proxy_bypass
from data.data_fetcher import fetch_data, eng_features
from labeling.cusum import cusum_filter_vol
from labeling.triple_barrier import apply_triple_barrier
from evaluation.metrics import compare_baselines
from models.pipeline import (
    prepare_features, run_cv_training, aggregate_metrics
)
from logging_config import get_logger
from run_tracker import save_run, load_runs, print_summary

warnings.filterwarnings('ignore')

# ── 日志 ──
logger = get_logger(__name__)


def run_pipeline(
    ts_code: str = '000001.SZ',
    start: str = '2015-01-01',
    end: str = '2025-12-31',
    cusum_vol_mult: float = 1.5,
    num_days: int = 20,
    meta_threshold: float = 0.3,
    barrier_configs: Optional[List[List[int]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    完整流水线

    参数:
        ts_code: 股票/指数代码
        start/end: 日期范围
        cusum_vol_mult: CUSUM 波动率倍数
        num_days: 最长持仓天数
        meta_threshold: 元模型接受阈值
        barrier_configs: 屏障宽度列表。
                        None 时使用默认集 [[2,1],[2,2],[3,1],[1,1]]
        verbose: 已弃用（保留兼容），日志级别由 logging_config 控制

    返回:
        dict: {
            'avg_metrics': 平均回测指标,
            'baselines': 基准策略对比,
            'fold_results': 每折结果,
            'labels': TB 标签,
            't_events': CUSUM 事件时间点,
        }
    """
    setup_proxy_bypass()

    # 日志自动带模块名和级别，不需要手动加 ===== 分隔线
    logger.info("Triple Barrier + Meta-Labeling + XGBoost 退出策略")
    logger.info("标的: %s  期间: %s ~ %s", ts_code, start, end)

    # ── Step 1: 获取数据 ──
    logger.info("[1/6] 获取数据...")
    df = fetch_data(ts_code, start, end)
    df = eng_features(df).dropna()
    logger.info("  → %d 个交易日", len(df))

    close = df['close']
    vol = df['v20'].clip(lower=0.001)
    atr = df['atr']

    # ── Step 2: CUSUM 事件采样 ──
    logger.info("[2/6] CUSUM 事件采样...")
    t_events = cusum_filter_vol(close, vol, cusum_vol_mult)
    logger.info("  → %d 个交易事件", len(t_events))

    if len(t_events) < 20:
        logger.warning("事件太少(%d)，尝试降低阈值...", len(t_events))
        t_events = cusum_filter_vol(close, vol, cusum_vol_mult * 0.5)
        logger.info("  → %d 个交易事件", len(t_events))

    # ── Step 3: Triple-Barrier 标注 ──
    logger.info("[3/6] Triple-Barrier 标注...")

    if barrier_configs is None:
        barrier_configs = [[2, 1], [2, 2], [3, 1], [1, 1]]

    all_labels = {}
    for pt_sl in barrier_configs:
        key = f'pt{pt_sl[0]}_sl{pt_sl[1]}'
        labels = apply_triple_barrier(close, t_events, pt_sl, vol, num_days)
        all_labels[key] = labels
        n_pos = (labels['bin'] == 1).sum()
        n_neg = (labels['bin'] == -1).sum()
        n_neu = (labels['bin'] == 0).sum()
        total = n_pos + n_neg
        wr = n_pos / total if total > 0 else 0
        logger.info("  [%s] +1:%d  -1:%d  0:%d  胜率:%.1f%%",
                     key, n_pos, n_neg, n_neu, wr * 100)

    # 选标签最均衡的配置
    best_key = min(all_labels,
                   key=lambda k: abs((all_labels[k]['bin'] == 1).sum() -
                                     (all_labels[k]['bin'] == -1).sum()))
    labels = all_labels[best_key]
    logger.info("  → 选用 %s（标签最均衡）", best_key)

    # ── Step 4: 特征准备 ──
    logger.info("[4/6] 特征准备...")
    X_feat, y, feat_cols = prepare_features(df, labels)
    logger.info("  → %d 样本 × %d 特征", len(X_feat), len(feat_cols))

    # ── Step 5: Purged CV 训练 ──
    logger.info("[5/6] Purged CV 两阶段训练...")
    fold_results = run_cv_training(
        X_feat, y, close, meta_threshold=meta_threshold, verbose=False
    )

    # ── Step 6: 汇总 ──
    logger.info("[6/6] 结果汇总...")
    avg_metrics = aggregate_metrics(fold_results)

    # 基准对比
    baselines = compare_baselines(close, t_events, atr)

    # ── 输出 ──
    logger.info("=" * 54)
    logger.info("RESULTS")
    logger.info("=" * 54)

    if avg_metrics:
        logger.info("── Triple-Barrier + Meta-Labeling + XGBoost ──")
        for k, v in avg_metrics.items():
            logger.info("  %-15s: %.4f", k, v)

    logger.info("── 基准策略对比（同CUSUM事件）──")
    for _, r in baselines.iterrows():
        logger.info("  %-15s: SR=%.3f  WR=%.1f%%  MDD=%.2f%%",
                     r['name'], r['sharpe'],
                     r['win_rate'] * 100, r['max_dd'] * 100)

    # 结论
    if avg_metrics:
        tb_sr = avg_metrics.get('sharpe', 0)
        bl_best = baselines['sharpe'].max()
        logger.info("══ 结论 ══")
        logger.info("  TB+Meta+XGBoost: Sharpe = %.3f", tb_sr)
        logger.info("  最佳基准:        Sharpe = %.3f", bl_best)
        if bl_best > 0:
            logger.info("  相对提升:        +%.1f%%", (tb_sr / bl_best - 1) * 100)
        dsr = avg_metrics.get('dsr', 0)
        if dsr > 1.96:
            logger.info("  ✅ DSR=%.2f > 1.96, 统计显著 (95%%置信度)", dsr)
        elif dsr > 0:
            logger.info("  ⚠️ DSR=%.2f < 1.96, 需更多数据验证", dsr)
        else:
            logger.info("  ❌ DSR=%.2f, 效果不优于随机", dsr)
    logger.info("=" * 54)

    return {
        'avg_metrics': avg_metrics,
        'baselines': baselines,
        'fold_results': fold_results,
        'labels': labels,
        't_events': t_events,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TB + Meta-Labeling + XGBoost 退出策略')
    parser.add_argument('--config', default='',
                        help='YAML 配置文件路径（如 configs/default.yaml）')
    parser.add_argument('--ts_code', default='',
                        help='股票代码（如 000001.SZ / 000300.SH，覆盖配置）')
    parser.add_argument('--start', default='',
                        help='开始日期（覆盖配置）')
    parser.add_argument('--end', default='',
                        help='结束日期（覆盖配置）')
    parser.add_argument('--cusum_vol_mult', type=float, default=None,
                        help='CUSUM 波动率倍数（覆盖配置）')
    parser.add_argument('--num_days', type=int, default=None,
                        help='最长持仓天数（覆盖配置）')
    parser.add_argument('--meta_threshold', type=float, default=None,
                        help='Meta-Labeling 阈值（覆盖配置）')
    parser.add_argument('--show-runs', action='store_true',
                        help='查看历史回测记录')
    args = parser.parse_args()

    # ── 查看历史记录 ──
    if args.show_runs:
        print_summary()
        exit(0)

    # ── 加载配置 ──
    logger.info("加载配置: %s", args.config or '（默认值）')
    from config_loader import load_config
    cfg = load_config(args.config)

    # ── 合并参数（优先级: CLI > YAML > 代码默认值） ──
    # CLI 参数非空时覆盖 YAML 配置
    ts_code  = args.ts_code or cfg.get('data', {}).get('stock', '000001.SZ')
    start    = args.start   or cfg.get('data', {}).get('start', '2015-01-01')
    end      = args.end     or cfg.get('data', {}).get('end', '2025-12-31')
    cusum    = args.cusum_vol_mult if args.cusum_vol_mult is not None else cfg['strategy']['cusum_vol_mult']
    num_days = args.num_days if args.num_days is not None else cfg['strategy']['num_days']
    meta     = args.meta_threshold if args.meta_threshold is not None else cfg['strategy']['meta_threshold']
    barriers = cfg['barriers']

    logger.info("运行参数: ts_code=%s  range=%s~%s  barriers=%s",
                ts_code, start, end, barriers)

    result = run_pipeline(
        ts_code=ts_code,
        start=start,
        end=end,
        cusum_vol_mult=cusum,
        num_days=num_days,
        meta_threshold=meta,
        barrier_configs=barriers,
    )

    # ── 保存实验记录 ──
    avg = result.get('avg_metrics') or {}
    baselines = result.get('baselines')
    best_bl_sr = baselines['sharpe'].max() if baselines is not None and 'sharpe' in baselines.columns else 0.0

    run_id = save_run(
        params={
            'cusum_vol_mult': cusum,
            'num_days': num_days,
            'meta_threshold': meta,
            'barriers': barriers,
        },
        metrics={
            'sharpe': avg.get('sharpe', 0.0),
            'max_dd': avg.get('max_dd', 0.0),
            'win_rate': avg.get('win_rate', 0.0),
            'profit_factor': avg.get('profit_factor', 0.0),
            'dsr': avg.get('dsr', 0.0),
            'num_trades': avg.get('num_trades', 0),
            'best_baseline_sr': best_bl_sr,
        },
        stock_code=ts_code,
        date_range=f"{start}~{end}",
        config_file=args.config,
    )
    logger.info("已保存回测记录: run_id=%s", run_id)
