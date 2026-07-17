"""
双阶段模型: Primary Model + Meta Model
支持 XGBoost / LightGBM / RandomForest 自动切换
"""
import numpy as np
import pandas as pd

# 尝试导入 XGBoost，失败则回退到 LightGBM 或 RandomForest
_BACKEND: str = 'xgboost'
_BaseClassifier = None  # type: ignore[assignment]
try:
    from xgboost import XGBClassifier
    _BaseClassifier = XGBClassifier
    _BACKEND = 'xgboost'
except ImportError:
    try:
        from lightgbm import LGBMClassifier
        _BaseClassifier = LGBMClassifier
        _BACKEND = 'lightgbm'
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        _BaseClassifier = RandomForestClassifier
        _BACKEND = 'randomforest'


def _make_classifier(**kwargs):
    """创建分类器，自动适配后端"""
    if _BACKEND == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**kwargs, verbosity=0, use_label_encoder=False,
                             eval_metric='logloss')
    elif _BACKEND == 'lightgbm':
        from lightgbm import LGBMClassifier
        kwargs.pop('verbosity', None)
        kwargs.pop('use_label_encoder', None)
        kwargs.pop('eval_metric', None)
        return LGBMClassifier(**kwargs, verbose=-1)
    else:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 4),
            random_state=42, n_jobs=-1)


def train_primary_model(X_train, y_train, params=None):
    """
    训练主模型：预测方向
    将三柱标签 {-1,0,1} 映射为二元分类: 1=做多, 0=不做
    """
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 4,
                  'learning_rate': 0.05, 'subsample': 0.8,
                  'colsample_bytree': 0.7, 'reg_alpha': 1.0,
                  'reg_lambda': 2.0, 'min_child_weight': 5,
                  'random_state': 42}
    y_binary = (y_train > 0).astype(int)
    model = _make_classifier(**params)
    model.fit(X_train, y_binary)
    return model


def train_meta_model(X_train, y_train, params=None):
    """
    训练元模型：预测信号是否可执行
    y_train: {0, 1}  1=可执行
    """
    if params is None:
        params = {'n_estimators': 80, 'max_depth': 3,
                  'learning_rate': 0.03, 'subsample': 0.7,
                  'colsample_bytree': 0.6, 'reg_alpha': 2.0,
                  'reg_lambda': 3.0, 'min_child_weight': 10,
                  'random_state': 42}
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    model = _make_classifier(**params)
    # 处理不平衡
    if hasattr(model, 'set_params') and pos > 0 and neg > 0:
        try:
            model.set_params(scale_pos_weight=neg / pos)
        except Exception:
            pass
    if hasattr(model, 'set_params') and hasattr(model, 'class_weight'):
        try:
            model.set_params(class_weight='balanced')
        except Exception:
            pass
    model.fit(X_train, y_train)
    return model


def predict_with_meta(primary_model, meta_model,
                      X_primary, X_meta,
                      meta_threshold=0.3,
                      sizing=False):
    """
    两阶段预测

    参数:
        primary_model: 主模型
        meta_model: 元模型
        X_primary: 主模型特征
        X_meta: 元模型特征
        meta_threshold: 元模型接受阈值
        sizing: 是否启用仓位调节
            - False: 返回 {-1, 0, 1} 信号（向后兼容）
            - True:  返回连续仓位 [-1.0, 1.0]
                      mp > 0.7  → 满仓 (±1.0)
                      mp 0.3-0.7 → 半仓 (±0.5) 
                      mp < 0.3  → 空仓 (0.0)

    返回:
        np.array 信号/仓位
    """
    # 主模型
    try:
        prob = primary_model.predict_proba(X_primary)
        if prob is not None and prob.ndim == 2 and prob.shape[1] >= 2:
            p1 = prob[:, 1]
        elif prob is not None and prob.ndim == 2 and prob.shape[1] == 1:
            p1 = prob[:, 0]
        else:
            p1 = np.full(len(X_primary), 0.5)
    except Exception:
        p1 = np.full(len(X_primary), 0.5)
    pred = (p1 > 0.5).astype(int)

    # 元模型
    try:
        meta_prob = meta_model.predict_proba(X_meta)
        if meta_prob is not None and meta_prob.ndim == 2 and meta_prob.shape[1] >= 2:
            mp = meta_prob[:, 1]
        elif meta_prob is not None and meta_prob.ndim == 2 and meta_prob.shape[1] == 1:
            mp = meta_prob[:, 0]
        else:
            mp = np.full(len(X_meta), 0.5)
    except Exception:
        mp = np.full(len(X_meta), 0.5)

    if not sizing:
        # 原始二值模式
        final = np.zeros(len(X_primary))
        buy = (mp > meta_threshold) & (p1 > 0.5)
        sell = (mp > meta_threshold) & (p1 <= 0.5)
        final[buy] = 1
        final[sell] = -1
    else:
        # 仓位调节模式
        direction = np.where(p1 > 0.5, 1, -1)  # 主模型方向
        final = np.zeros(len(X_primary))

        # 高置信度 → 满仓
        high_conf = mp > 0.7
        final[high_conf] = direction[high_conf]

        # 中置信度 → 半仓
        mid_conf = (mp >= meta_threshold) & (mp <= 0.7)
        final[mid_conf] = direction[mid_conf] * 0.5

        # 低置信度 → 空仓（保持 0）

    return final


def generate_meta_features(X_primary, primary_model):
    """
    为元模型构造特征 = 主模型输出 + 市场状态
    """
    try:
        prob = primary_model.predict_proba(X_primary)
        if prob is not None and prob.ndim == 2 and prob.shape[1] >= 2:
            p1 = prob[:, 1]
            margin = prob[:, 1] - prob[:, 0]
        elif prob is not None and prob.ndim == 2 and prob.shape[1] == 1:
            p1 = prob[:, 0]
            margin = np.zeros(len(X_primary))
        else:
            p1 = np.full(len(X_primary), 0.5)
            margin = np.zeros(len(X_primary))
    except Exception:
        p1 = np.full(len(X_primary), 0.5)
        margin = np.zeros(len(X_primary))

    try:
        pred = primary_model.predict(X_primary)
    except Exception:
        pred = (p1 > 0.5).astype(int)

    meta = pd.DataFrame(index=X_primary.index)
    meta['primary_signal'] = np.where(pred == 1, 1, -1)
    meta['primary_confidence'] = np.maximum(p1, 1 - p1)
    meta['primary_margin'] = margin

    for col in ['v10', 'vv', 'atr_pct', 'bbp', 'rsi',
                'vr', 'closepos', 'r20', 'brk_h', 'brk_l']:
        if col in X_primary.columns:
            meta[col] = X_primary[col].values
    return meta
