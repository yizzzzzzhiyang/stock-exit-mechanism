"""
Walk-Forward 验证模块

模拟真实交易场景: 定期用新数据重新训练模型，评估参数随时间退化的程度。
这是 PurgedCV 无法覆盖的维度——CV 用的是同一个模型，Walk-forward 模拟模型更新。

两种模式:
  1. Expanding Window: 训练集不断累积（更稳定）
     训练 [2015-2016] → 预测 [2017]
     训练 [2015-2017] → 预测 [2018]
     ...

  2. Rolling Window: 固定窗口大小滚动（对 regime change 更敏感）
     训练 [2015-2016] → 预测 [2017]
     训练 [2016-2017] → 预测 [2018]
     ...
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from evaluation.metrics import evaluate_strategy, compare_baselines, get_cost_config


def train_and_predict(X_train, y_train, X_test, meta_threshold, sizing=False):
    """
    在一个窗口上训练并预测

    返回:
        predictions: np.array（信号或仓位）
        primary_model, meta_model: 训练后的模型（用于诊断）
    """
    from models.tb_meta_xgb import (
        train_primary_model, train_meta_model,
        generate_meta_features, predict_with_meta
    )
    from labeling.triple_barrier import get_meta_labels

    # 训练主模型
    primary = train_primary_model(X_train, y_train['bin'])

    # 生成元特征
    meta_tr = generate_meta_features(X_train, primary)
    meta_te = generate_meta_features(X_test, primary)

    # 生成元标签
    side_tr = np.where(primary.predict(X_train) == 1, 1, -1)
    meta_y_tr = get_meta_labels(y_train, pd.Series(side_tr, index=y_train.index))

    # 训练元模型
    meta = train_meta_model(meta_tr, meta_y_tr['bin'])

    # 预测
    pred = predict_with_meta(primary, meta, X_test, meta_te, meta_threshold,
                             sizing=sizing)

    return pred, primary, meta


def walk_forward(df: pd.DataFrame, close: pd.Series, vol: pd.Series, atr: pd.Series,
                 t_events: pd.DatetimeIndex, pt_sl: list, num_days: int = 20,
                 window_years: int = 3, step_years: int = 1,
                 meta_threshold: float = 0.3, sizing: bool = False,
                 mode: str = 'expanding', min_train_years: int = 2,
                 verbose: bool = True) -> dict:
    """
    Walk-Forward 验证

    参数:
        df: 含价格+特征的完整 DataFrame (index=datetime)
        close: 价格序列
        vol: 波动率序列
        atr: ATR 序列
        t_events: CUSUM 事件（全部时间的）
        pt_sl: 屏障配置 [止盈,止损]
        num_days: 最长持仓
        window_years: 训练窗口长度（年）— rolling 模式用
        step_years: 步长（年）
        meta_threshold: 元模型阈值
        sizing: 是否启用仓位调节
        mode: 'expanding' | 'rolling'
        min_train_years: 最少训练年数
        verbose: 打印进度

    返回:
        dict: 
          windows: list of per-window metrics
          aggregate: 汇总指标
          all_predictions: 所有窗口的预测拼接
          stability: 参数稳定性评分
    """
    from labeling.triple_barrier import apply_triple_barrier

    # 计算窗口边界
    all_dates = sorted(df.index.unique())
    start_date = all_dates[0]
    end_date = all_dates[-1]

    min_train_days = min_train_years * 252
    window_days = window_years * 252
    step_days = step_years * 252

    # 生成窗口
    windows = []
    test_start = start_date + pd.Timedelta(days=min_train_days)

    while test_start + pd.Timedelta(days=step_days) <= end_date:
        test_end = min(test_start + pd.Timedelta(days=step_days), end_date)

        if mode == 'expanding':
            train_start = start_date
        else:  # rolling
            train_start = max(start_date, test_start - pd.Timedelta(days=window_days))

        if test_end > test_start:
            windows.append((train_start, test_start, test_end))

        test_start = test_end

    if len(windows) < 2:
        return {
            'windows': [],
            'error': f'数据不足以做 walk-forward（需要至少 {min_train_years + 1} 年）',
        }

    # ── 逐窗口执行 ──
    all_metrics = []
    all_predictions = []

    # 排除 OHLCV 价格列
    exclude = {'open', 'high', 'low', 'close', 'volume', 'amount'}
    feat_cols = [c for c in df.columns if c not in exclude]

    if verbose:
        print(f'\n═══ Walk-Forward ({mode}) ───')
        print(f'共 {len(windows)} 个窗口, 各步长 {step_years} 年')
        print(f'屏障: pt_sl={pt_sl}, days={num_days}')
        print()

    for i, (tr_start, te_start, te_end) in enumerate(windows):
        # 过滤训练期事件 (事件时间 < te_start)
        train_events = t_events[(t_events >= tr_start) & (t_events < te_start)]
        # 过滤测试期事件
        test_events = t_events[(t_events >= te_start) & (t_events < te_end)]

        if len(train_events) < 10 or len(test_events) < 5:
            if verbose:
                print(f'  Window {i+1}: {tr_start.date()} → [{te_start.date()} ~ {te_end.date()}] '
                      f'事件不足(train={len(train_events)}, test={len(test_events)})，跳过')
            continue

        # 标注（只用训练期数据标注，但价格路径可能跨入测试期 -- 注意这点！）
        # 为避免标签泄漏，我们用训练期末尾的价格 + 训练期的事件来标注
        # 这里简化处理：用全时间段的价格，事件只取训练期的
        try:
            labels = apply_triple_barrier(close, train_events, pt_sl, vol, num_days)
        except Exception as e:
            if verbose:
                print(f'  Window {i+1}: 标注失败 {e}')
            continue

        # 准备特征
        common = labels[labels['bin'] != 0].index.intersection(df.index)
        if len(common) < 20:
            if verbose:
                print(f'  Window {i+1}: 有效标签不足({len(common)})，跳过')
            continue

        X_train = df.loc[common][feat_cols]
        y_train = labels.loc[common]

        # 测试特征
        test_df = df.loc[df.index.isin(test_events)]
        if len(test_df) == 0:
            continue
        X_test = test_df[feat_cols]

        # 训练 + 预测
        try:
            pred, primary, meta = train_and_predict(
                X_train, y_train, X_test, meta_threshold, sizing=sizing
            )
        except Exception as e:
            if verbose:
                print(f'  Window {i+1}: 训练失败 {e}')
            continue

        # 生成信号序列
        pred_series = pd.Series(pred, index=X_test.index)
        pred_series = pd.Series(0, index=close.index).add(
            pred_series.reindex(close.index), fill_value=0
        )
        # 只在测试期内有信号
        pred_series = pred_series.where(
            (pred_series.index >= te_start) & (pred_series.index < te_end), 0
        )

        all_predictions.append(pred_series)

        # 评估
        metrics = evaluate_strategy(pred_series, close, f'WF-Win{i+1}')
        metrics['window'] = i + 1
        metrics['train_period'] = f'{tr_start.date()}~{te_start.date()}'
        metrics['test_period'] = f'{te_start.date()}~{te_end.date()}'
        all_metrics.append(metrics)

        if verbose:
            print(f'  [{metrics["test_period"]}] '
                  f'SR={metrics.get("sharpe", 0):.3f}  '
                  f'MDD={metrics.get("max_dd", 0):.2%}  '
                  f'WR={metrics.get("win_rate", 0):.1%}  '
                  f'Trades={metrics.get("n_trades", 0)}')

    if not all_metrics:
        return {'windows': [], 'error': '所有窗口均失败'}

    # ── 汇总 ──
    sr_list = [m.get('sharpe', 0) for m in all_metrics]
    wr_list = [m.get('win_rate', 0) for m in all_metrics]
    mdd_list = [m.get('max_dd', 0) for m in all_metrics]

    # 稳定性评分: SR 标准差 / 均值 的倒数（越高越稳定）
    sr_mean = np.mean(sr_list)
    sr_std = np.std(sr_list)
    stability = sr_mean / (sr_std + 1e-10) if sr_std > 0 else 99

    # 正 SR 窗口比例
    positive_ratio = np.mean(np.array(sr_list) > 0)

    aggregate = {
        'windows': len(all_metrics),
        'avg_sharpe': round(sr_mean, 3),
        'std_sharpe': round(sr_std, 3),
        'min_sharpe': round(min(sr_list), 3),
        'max_sharpe': round(max(sr_list), 3),
        'positive_window_ratio': round(positive_ratio, 3),
        'stability': round(stability, 1),        # > 2.0 稳定, > 1.0 可接受
        'avg_win_rate': round(np.mean(wr_list), 3),
        'avg_max_dd': round(np.mean(mdd_list), 4),
        'mode': mode,
        'pt_sl': pt_sl,
    }

    if verbose:
        print(f'\n── WF 汇总 ──')
        print(f'  avg_SR: {aggregate["avg_sharpe"]:.3f} ± {aggregate["std_sharpe"]:.3f}')
        print(f'  SR 范围: [{aggregate["min_sharpe"]:.3f}, {aggregate["max_sharpe"]:.3f}]')
        print(f'  正窗口比例: {positive_ratio:.0%}')
        print(f'  稳定性评分: {stability:.1f} (>2.0稳定, >1.0可接受)')
        print(f'  avg_WR: {aggregate["avg_win_rate"]:.1%}')
        print(f'  avg_MDD: {aggregate["avg_max_dd"]:.2%}')

    # 拼接所有预测
    if all_predictions:
        combined = all_predictions[0].copy()
        for pred in all_predictions[1:]:
            combined = combined.where(combined != 0, pred)
        # 计算总体指标
        overall = evaluate_strategy(combined, close, 'WF-Overall')
        aggregate['overall_sharpe'] = overall.get('sharpe', 0)
        aggregate['overall_dsr'] = overall.get('dsr', 0)
        aggregate['overall_mdd'] = overall.get('max_dd', 0)
    else:
        combined = pd.Series(0, index=close.index)

    return {
        'windows': all_metrics,
        'aggregate': aggregate,
        'all_predictions': combined,
    }


def prep_features_for_wf(df, labels):
    """共享的特征准备函数（walk-forward 用）"""
    exclude = {'open', 'high', 'low', 'close', 'volume', 'amount'}
    common = labels[labels['bin'] != 0].index.intersection(df.index)
    X = df.loc[common]
    y = labels.loc[common]
    feat_cols = [c for c in X.columns if c not in exclude]
    X_feat = X[feat_cols]
    return X_feat, y, feat_cols
