"""
共享训练流水线 — main.py 和 optimized.py 共用

抽取核心逻辑:
  1. prepare_features: 特征准备（对齐标签、过滤列）
  2. run_fold: 单 fold 训练+预测
  3. run_cv_training: 完整 PurgedCV 训练循环
  4. run_threshold_search: 多阈值搜索（optimized 专用）
"""
import numpy as np
import pandas as pd

from cv.purged_cv import PurgedCV
from models.tb_meta_xgb import (
    train_primary_model, train_meta_model,
    generate_meta_features, predict_with_meta
)
from labeling.triple_barrier import get_meta_labels
from evaluation.metrics import evaluate_strategy


def prepare_features(df: pd.DataFrame, labels: pd.DataFrame):
    """
    特征准备: 对齐 DataFrame 和标签，过滤非特征列

    参数:
        df: 含价格+特征的完整 DataFrame
        labels: Triple-Barrier 标注结果（含 'bin' 列）

    返回:
        X_feat: 纯特征 DataFrame
        y: 对齐后的标签
        feat_cols: 特征列名列表
    """
    # 只保留有方向性标签的样本（bin != 0）
    common = labels[labels['bin'] != 0].index.intersection(df.index)
    X = df.loc[common]
    y = labels.loc[common]

    # 排除 OHLCV 价格列
    exclude = {'open', 'high', 'low', 'close', 'volume', 'amount'}
    feat_cols = [c for c in X.columns if c not in exclude]
    X_feat = X[feat_cols]

    return X_feat, y, feat_cols


def run_fold(X_tr, X_te, y_tr, y_te, close, meta_threshold, fold_name):
    """
    单 fold 训练+评估

    返回: metrics dict 或 None（fold 失败时）
    """
    if len(X_tr) < 50 or len(set((y_tr['bin'] > 0).astype(int))) < 2:
        return None

    # 1) 训练主模型
    primary = train_primary_model(X_tr, y_tr['bin'])

    # 2) 生成元特征
    meta_tr = generate_meta_features(X_tr, primary)
    meta_te = generate_meta_features(X_te, primary)

    # 3) 生成元标签
    side_tr = np.where(primary.predict(X_tr) == 1, 1, -1)
    side_te = np.where(primary.predict(X_te) == 1, 1, -1)
    meta_y_tr = get_meta_labels(y_tr, pd.Series(side_tr, index=y_tr.index))
    meta_y_te = get_meta_labels(y_te, pd.Series(side_te, index=y_te.index))

    # 4) 训练元模型
    meta_model = train_meta_model(meta_tr, meta_y_tr['bin'])

    # 5) 预测
    final_pred = predict_with_meta(
        primary, meta_model, X_te, meta_te, meta_threshold
    )

    # 6) 评估
    pred_series = pd.Series(final_pred, index=X_te.index)
    metrics = evaluate_strategy(pred_series, close, fold_name)
    return metrics


def run_cv_training(X_feat, y, close, meta_threshold=0.3,
                    n_splits=4, embargo=5, verbose=True):
    """
    完整 PurgedCV 训练循环

    参数:
        X_feat: 纯特征 DataFrame
        y: 标签（含 'bin' 和 't1' 列）
        close: 价格序列
        meta_threshold: 元模型阈值
        n_splits: CV 折数
        embargo: 训练/测试间隔天数
        verbose: 是否打印每折结果

    返回:
        fold_results: list of metrics dict
    """
    cv = PurgedCV(n_splits=n_splits, embargo=embargo)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_feat, y)):
        X_tr = X_feat.iloc[train_idx]
        X_te = X_feat.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        try:
            metrics = run_fold(
                X_tr, X_te, y_tr, y_te,
                close, meta_threshold, f'Fold{fold+1}'
            )
            if metrics:
                fold_results.append(metrics)
                if verbose:
                    print(
                        f'      Fold {fold+1}/{n_splits}: '
                        f'SR={metrics["sharpe"]:.3f}  '
                        f'DSR={metrics["dsr"]:.3f}  '
                        f'WR={metrics["win_rate"]:.1%}  '
                        f'Trades={metrics["n_trades"]}  '
                        f'PF={metrics["profit_factor"]:.2f}'
                    )
        except Exception as e:
            if verbose:
                print(f'      Fold {fold+1}/{n_splits}: 错误 {e}')

    return fold_results


def run_threshold_search(X_feat, y, close, thresholds=None,
                         n_splits=4, embargo=5, verbose=True):
    """
    多阈值搜索 — 找出最优 meta_threshold

    参数:
        X_feat: 特征
        y: 标签
        close: 价格
        thresholds: 候选阈值列表
        n_splits, embargo: CV 参数
        verbose: 打印搜索过程

    返回:
        (best_threshold, all_results)
        all_results: [(threshold, avg_sharpe, valid_folds, fold_metrics), ...]
    """
    if thresholds is None:
        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    cv = PurgedCV(n_splits=n_splits, embargo=embargo)
    best_threshold = 0.3
    best_score = -999
    all_results = []

    for th in thresholds:
        fold_metrics = []
        valid_folds = 0
        for fold, (trn, tst) in enumerate(cv.split(X_feat, y)):
            X_tr = X_feat.iloc[trn]
            X_te = X_feat.iloc[tst]
            y_tr = y.iloc[trn]
            y_te = y.iloc[tst]
            try:
                metrics = run_fold(
                    X_tr, X_te, y_tr, y_te,
                    close, th, f'F{fold+1}'
                )
                if metrics:
                    fold_metrics.append(metrics)
                    valid_folds += 1
            except Exception:
                pass

        if valid_folds >= 2:
            avg_sr = np.mean([m['sharpe'] for m in fold_metrics])
            all_results.append((th, avg_sr, valid_folds, fold_metrics))
            if avg_sr > best_score:
                best_score = avg_sr
                best_threshold = th

    if verbose:
        print(f'\n阈值搜索:')
        for th, sr, nf, _ in sorted(all_results, key=lambda x: -x[1])[:5]:
            print(f'  th={th:.2f}  avg_SR={sr:.3f}  folds={nf}')
        print(f'最佳阈值: th={best_threshold:.2f}')

    return best_threshold, all_results


def aggregate_metrics(fold_results: list) -> dict:
    """汇总各 fold 的指标为平均值"""
    if not fold_results:
        return {}
    avg = {}
    for key in ['sharpe', 'dsr', 'sortino', 'max_dd',
                'calmar', 'win_rate', 'profit_factor']:
        vals = [r.get(key, 0) for r in fold_results if key in r]
        if vals:
            avg[key] = float(np.mean(vals))
    return avg
