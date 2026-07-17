"""
特征分析: 共线性检测 + SHAP 重要性排序

功能:
  1. corr_analysis: 特征相关性矩阵，标记高相关对
  2. shap_importance: SHAP 特征重要性排序（需训练 XGBoost）
  3. suggest_drop: 基于共线性+SHAP 推荐可剔除的冗余特征
  4. plot_corr_heatmap: ASCII 相关性热力图
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def corr_analysis(X: pd.DataFrame, threshold: float = 0.85) -> dict:
    """
    特征共线性分析

    参数:
        X: 特征 DataFrame
        threshold: 相关性阈值，超过此值的对视为高相关

    返回:
        dict:
          high_corr_pairs: [(feat1, feat2, corr), ...] 高相关对
          corr_matrix: 相关性矩阵
          redundancy_score: 每个特征的冗余分数
    """
    corr = X.corr()
    n = len(corr.columns)

    # 找出上三角中超过阈值的对
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                high_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

    high_pairs.sort(key=lambda x: -abs(x[2]))

    # 冗余分数: 每个特征与其他所有特征的平均绝对相关性
    redundancy = {}
    for col in corr.columns:
        others = [c for c in corr.columns if c != col]
        redundancy[col] = round(float(corr.loc[col, others].abs().mean()), 3)

    return {
        'high_corr_pairs': high_pairs,
        'corr_matrix': corr,
        'redundancy_score': redundancy,
    }


def shap_importance(X: pd.DataFrame, y: pd.Series, max_display: int = 20) -> dict:
    """
    SHAP 特征重要性分析

    参数:
        X: 特征 DataFrame
        y: 目标变量 (0/1 二分类)
        max_display: 显示前 N 个重要特征

    返回:
        dict:
          top_features: [(feature, shap_importance), ...]
          shap_values: numpy array (可选)
    """
    try:
        from xgboost import XGBClassifier
        import shap
    except ImportError:
        return {
            'top_features': [],
            'error': 'shap 未安装。pip install shap 后重试。'
        }

    # 训练轻量 XGBoost
    model = XGBClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        verbosity=0, use_label_encoder=False, eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)

    # SHAP 值
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # fallback: use feature_importances_
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top = [(X.columns[i], round(float(importances[i]), 6))
               for i in sorted_idx[:max_display]]
        return {
            'top_features': top,
            'method': 'feature_importances_ (SHAP 计算失败，回退到内置重要性)',
        }

    # 汇总为特征重要性
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top = [(X.columns[i], round(float(mean_abs_shap[i]), 6))
           for i in sorted_idx[:max_display]]

    return {
        'top_features': top,
        'shap_values': shap_values,
        'method': 'SHAP',
    }


def suggest_drop(X: pd.DataFrame, y: pd.Series = None,
                 corr_threshold: float = 0.85) -> dict:
    """
    综合共线性 + SHAP，推荐可剔除的冗余特征

    规则:
      1. 如果两个特征相关系数 > corr_threshold, 且其中一个 SHAP 重要度低,
         建议保留 SHAP 高者，剔除低者
      2. 如果没有 SHAP (无标签), 只按共线性推荐（保守剔除）

    参数:
        X: 特征 DataFrame
        y: 目标变量（可选，用于 SHAP）
        corr_threshold: 相关性阈值

    返回:
        dict:
          to_drop: [feature_name, ...] 建议剔除的特征
          to_keep: [feature_name, ...] 建议保留的特征
          reasoning: [{pair, drop, keep, reason}, ...]
    """
    corr_result = corr_analysis(X, threshold=corr_threshold)

    # 尝试获取 SHAP 重要性
    shap_result = None
    shap_rank = {}
    if y is not None:
        try:
            shap_result = shap_importance(X, y)
            for i, (feat, score) in enumerate(shap_result.get('top_features', [])):
                shap_rank[feat] = i
        except Exception:
            pass

    reasoning = []
    to_drop = set()
    to_keep = set()

    for feat1, feat2, corr_val in corr_result['high_corr_pairs']:
        if shap_rank:
            rank1 = shap_rank.get(feat1, 999)
            rank2 = shap_rank.get(feat2, 999)
            if rank1 < rank2:
                # feat1 更重要
                drop_feat, keep_feat = feat2, feat1
            else:
                drop_feat, keep_feat = feat1, feat2
            reason = f'共线(r={corr_val}) | SHAP 排名: {keep_feat}(#{rank1+1}) > {drop_feat}(#{rank2+1})'
        else:
            # 无 SHAP: 保守地剔除绝对值较大的特征（保留与其他特征平均相关度更低的）
            red1 = corr_result['redundancy_score'].get(feat1, 0)
            red2 = corr_result['redundancy_score'].get(feat2, 0)
            if red1 <= red2:
                drop_feat, keep_feat = feat2, feat1
            else:
                drop_feat, keep_feat = feat1, feat2
            reason = f'共线(r={corr_val}) | 冗余度: {keep_feat}({red1}) < {drop_feat}({red2})'

        to_drop.add(drop_feat)
        to_keep.add(keep_feat)
        reasoning.append({
            'pair': (feat1, feat2),
            'corr': corr_val,
            'drop': drop_feat,
            'keep': keep_feat,
            'reason': reason,
        })

    return {
        'to_drop': sorted(to_drop),
        'to_keep': sorted(to_keep),
        'reasoning': reasoning,
    }


def print_feature_report(X: pd.DataFrame, y: pd.Series = None):
    """打印完整的特征分析报告"""
    print('=' * 60)
    print('特征分析报告')
    print('=' * 60)

    # 1) 基本统计
    print(f'\n特征总数: {len(X.columns)}')
    print(f'样本数:   {len(X)}')

    # 2) 共线性
    corr_result = corr_analysis(X)
    high_pairs = corr_result['high_corr_pairs']
    print(f'\n── 共线性分析 (|r| >= 0.85) ──')
    if high_pairs:
        print(f'发现 {len(high_pairs)} 对高相关特征:')
        for f1, f2, r in high_pairs[:15]:
            print(f'  {f1:20s} ↔ {f2:20s}  r={r:.3f}')
        if len(high_pairs) > 15:
            print(f'  ... 还有 {len(high_pairs) - 15} 对')
    else:
        print('  ✅ 无高相关特征对')

    # 3) 冗余度 top-10
    red = corr_result['redundancy_score']
    top_red = sorted(red.items(), key=lambda x: -x[1])[:10]
    print(f'\n── 冗余度 Top-10（平均 |r| 最高）──')
    for feat, score in top_red:
        print(f'  {feat:20s}: {score:.3f}')

    # 4) SHAP
    if y is not None:
        print(f'\n── SHAP 重要性 Top-20 ──')
        shap_result = shap_importance(X, y)
        if 'error' in shap_result:
            print(f'  ⚠ {shap_result["error"]}')
        else:
            print(f'  方法: {shap_result.get("method", "SHAP")}')
            for feat, score in shap_result['top_features'][:20]:
                bar = '█' * min(int(score * 5000), 30)
                print(f'  {feat:20s}: {score:.6f} {bar}')

    # 5) 剔除建议
    if y is not None:
        drop_result = suggest_drop(X, y)
        print(f'\n── 建议剔除（共线 + SHAP 低重要度）──')
        if drop_result['to_drop']:
            print(f'  建议剔除 {len(drop_result["to_drop"])} 个特征:')
            for f in drop_result['to_drop']:
                print(f'    ✗ {f}')
            for r in drop_result['reasoning'][:10]:
                print(f'    {r["reason"]}')
        else:
            print('  ✅ 无建议剔除的特征')

    print('\n' + '=' * 60)
