"""
Purged Cross-Validation（去泄露交叉验证）

金融时序不是 IID 的，标准 K-Fold CV 会泄露未来信息。

本模块实现:
  1. Purging: 从训练集中删除与测试集有时间重叠的样本
  2. Embargo: 在训练集和测试集之间留出间隔
  3. Nested CV: 外层评估泛化能力，内层选择超参

这是防止三柱法标签泄漏的最关键模块。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedCV(BaseCrossValidator):
    """
    去泄露时间序列交叉验证

    参数:
        n_splits: 折数
        embargo: 训练/测试间隔（天），防止相邻样本泄漏
        purge: 是否清除与测试集时间重叠的训练样本
    """

    def __init__(self, n_splits=5, embargo=5, purge=True):
        self.n_splits = n_splits
        self.embargo = pd.Timedelta(days=embargo)
        self.purge = purge
        self._test_indices = []
        self._train_indices = []

    def split(self, X, y=None, groups=None):
        """生成(train_idx, test_idx)对"""
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = n if i == self.n_splits - 1 else (i + 1) * fold_size
            test_idx = indices[test_start:test_end]

            # 训练集: 测试集之前的所有数据
            train_idx = indices[:test_start]

            # Embargo: 测试集之前的 embargo 天也要排除
            if self.embargo:
                embargo_end = test_start
                # 找到 embargo_end 对应的位置往前推
                embargo_start = max(0, test_start - self.embargo.days)
                # 但 embargo_start 不能是负数索引
                # 我们用更精确的方式: 找时间戳
                if hasattr(X, 'index'):
                    test_t0 = X.index[test_start] if hasattr(X, 'index') else None
                    if test_t0 is not None:
                        embargo_cutoff = test_t0 - self.embargo
                        embargo_mask = X.index[:test_start] >= embargo_cutoff
                        train_idx = indices[:test_start][~embargo_mask]

            # Purge: 清除训练集中 t1 超过测试集起始时间的样本
            if self.purge and hasattr(X, 'index'):
                # 对于三柱法标签, t1 在 triple_barrier_result['t1'] 中
                # 如果 y 是 DataFrame 且含 't1' 列
                if isinstance(y, pd.DataFrame) and 't1' in y.columns:
                    test_start_date = X.index[test_start]
                    leaked = []
                    for j, t_idx in enumerate(train_idx):
                        t1_date = y['t1'].iloc[t_idx]
                        if pd.notna(t1_date) and t1_date >= test_start_date:
                            leaked.append(j)
                    if leaked:
                        train_idx = np.delete(train_idx, leaked)

            self._train_indices.append(train_idx)
            self._test_indices.append(test_idx)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class NestedPurgedCV:
    """
    嵌套 Purged CV:
      外层 CV: 评估模型泛化能力
      内层 CV: 选择最优超参（屏障宽度 + 模型参数）

    这是防止"用测试集信息选参数"的核心方案
    """

    def __init__(self, outer_splits=5, inner_splits=3,
                 embargo=5, purge=True):
        self.outer_cv = PurgedCV(n_splits=outer_splits, embargo=embargo, purge=purge)
        self.inner_cv = PurgedCV(n_splits=inner_splits, embargo=embargo, purge=purge)

    def search_params(self, X, y, param_grid, train_model_fn,
                      score_fn, verbose=True):
        """
        嵌套超参搜索

        参数:
            X: 特征
            y: 标签 (含 t1 列用于 purging)
            param_grid: 超参列表 [{'pt_sl':[2,2], 'max_depth':3, ...}, ...]
            train_model_fn: 函数 fn(X_train, y_train, params) → model
            score_fn: 函数 fn(model, X_val, y_val) → score

        返回:
            best_params, cv_scores
        """
        outer_scores = []
        all_results = []

        for fold, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 内层 CV 搜索
            best_inner_score = -np.inf
            best_inner_params = param_grid[0]

            for params in param_grid:
                inner_scores = []
                for inner_train, inner_val in self.inner_cv.split(X_train, y_train):
                    model = train_model_fn(X_train.iloc[inner_train],
                                           y_train.iloc[inner_train],
                                           params)
                    score = score_fn(model, X_train.iloc[inner_val],
                                     y_train.iloc[inner_val])
                    inner_scores.append(score)
                avg_score = np.mean(inner_scores)
                if avg_score > best_inner_score:
                    best_inner_score = avg_score
                    best_inner_params = params

            # 用最优参数在外层测试集上评估
            best_model = train_model_fn(X_train, y_train, best_inner_params)
            test_score = score_fn(best_model, X_test, y_test)
            outer_scores.append(test_score)
            all_results.append({
                'fold': fold,
                'best_params': best_inner_params,
                'test_score': test_score,
                'inner_best_score': best_inner_score,
            })

        return all_results, np.mean(outer_scores), np.std(outer_scores)
