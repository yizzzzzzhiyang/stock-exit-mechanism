"""
CUSUM 滤波器 —— 事件驱动的采样
价格累计变化超过阈值时才产生交易信号
避免在平淡行情中过度采样
"""
import numpy as np
import pandas as pd


def cusum_filter(close: pd.Series, threshold: float = 0.02,
                 return_events: bool = True) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter
    参数:
        close: 价格序列
        threshold: 阈值（百分比），例如 0.02 = 2%
        return_events: True 返回事件时间戳列表，False 返回累计和序列
    返回:
        pd.DatetimeIndex 事件发生的时间戳
    """
    if isinstance(threshold, pd.Series):
        threshold = threshold.iloc[0] if len(threshold) > 0 else 0.02
    t_events = []
    s_pos, s_neg = 0.0, 0.0
    diff = close.diff().values
    idx = close.index

    for i in range(1, len(close)):
        s_pos = max(0, s_pos + diff[i])
        s_neg = min(0, s_neg + diff[i])
        # 用 threshold * close[i] 作为绝对阈值，适应价格水平
        th = threshold * close.iloc[i]
        if s_pos > th:
            s_pos = 0
            t_events.append(idx[i])
        elif s_neg < -th:
            s_neg = 0
            t_events.append(idx[i])

    return pd.DatetimeIndex(t_events)


def cusum_filter_vol(close: pd.Series, vol_pct: pd.Series,
                     vol_mult: float = 1.5) -> pd.DatetimeIndex:
    """
    基于波动率的 CUSUM——用百分比收益计算
    vol_pct: 百分比波动率（如 0.02 = 2%）
    阈值 = vol_mult × vol_pct（百分比级别，无量纲）
    """
    t_events = []
    s_pos, s_neg = 0.0, 0.0
    # 用百分比收益代替绝对价格差
    ret = close.pct_change().values
    idx = close.index
    vol_vals = vol_pct.values if hasattr(vol_pct, 'values') else np.full(len(close), 0.02)

    for i in range(1, len(close)):
        s_pos = max(0, s_pos + ret[i])
        s_neg = min(0, s_neg + ret[i])
        th = vol_mult * vol_vals[min(i, len(vol_vals)-1)] if i < len(vol_vals) else vol_mult * 0.02
        if s_pos > th:
            s_pos = 0
            t_events.append(idx[i])
        elif s_neg < -th:
            s_neg = 0
            t_events.append(idx[i])

    return pd.DatetimeIndex(t_events)
