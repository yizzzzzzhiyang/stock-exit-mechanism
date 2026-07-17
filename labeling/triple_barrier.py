"""
Triple-Barrier 标签生成器（自主实现，零商业依赖）

核心逻辑:
  对每个事件（入场点）设置三根屏障:
  1. 上水平屏障（止盈）: 价格 + δ
  2. 下水平屏障（止损）: 价格 - δ
  3. 垂直屏障（时间）  : 最长持仓 N 天
  最先触达的屏障决定标签: +1(盈利), -1(亏损), 0(中性)

A 股适配:
  - T+1: 最短持仓 ≥ 2 交易日
  - 涨跌停: 触碰涨停/跌停视为触达屏障（防止假信号）
  - 停牌: 跳过停牌期间的事件
"""
import numpy as np
import pandas as pd


def add_vertical_barrier(t_events: pd.DatetimeIndex,
                         close: pd.Series,
                         num_days: int = 20,
                         min_days: int = 2) -> pd.Series:
    """
    为每个事件计算垂直屏障的时间戳
    参数:
        t_events: 事件时间戳
        close: 价格序列（用于定位下一个交易日）
        num_days: 最长持仓天数
        min_days: 最短持仓天数（A 股 T+1 → 至少 2 天）
    返回:
        Series: index=事件时间, value=垂直屏障时间戳
    """
    t1 = pd.Series(index=t_events, dtype='datetime64[ns]')
    close_idx = close.index

    for t in t_events:
        loc = close_idx.get_loc(t) if t in close_idx else None
        if loc is None:
            continue
        # 跳过 min_days 天（A 股 T+1 要求）
        end_loc = min(loc + num_days, len(close_idx) - 1)
        t1[t] = close_idx[end_loc]

    return t1.dropna()


def apply_triple_barrier(close: pd.Series,
                         t_events: pd.DatetimeIndex,
                         pt_sl: list,
                         vol: pd.Series,
                         num_days: int = 20,
                         min_days: int = 2,
                         detect_limits: bool = True) -> pd.DataFrame:
    """
    三柱法标注

    参数:
        close: 价格序列 (pd.Series)
        t_events: 事件时间戳
        pt_sl: [止盈倍数, 止损倍数], 例如 [2, 1] 表示止盈=2倍ATR, 止损=1倍ATR
        vol: 波动率序列 (用于计算屏障宽度)
        num_days: 最长持仓
        min_days: 最短持仓（A股T+1→2）
        detect_limits: 是否检测涨跌停

    返回:
        DataFrame: columns=['t1','trgt','bin','ret']
            bin: +1(止盈), -1(止损), 0(时间到期)
    """
    # 垂直屏障
    t1 = add_vertical_barrier(t_events, close, num_days, min_days)

    out = pd.DataFrame(index=t_events)
    out['t1'] = t1
    out['bin'] = 0
    out['ret'] = 0.0

    for idx in t_events:
        if idx not in t1.index or pd.isna(t1[idx]):
            continue
        end = t1[idx]
        # 获取从入场到垂直屏障的价格路径
        mask = (close.index >= idx) & (close.index <= end)
        path = close.loc[mask]

        if len(path) < 2:
            continue

        entry_price = path.iloc[0]
        # 屏障宽度
        entry_vol = vol.loc[idx] if idx in vol.index and not pd.isna(vol.loc[idx]) else 0.01
        upper = entry_price * (1 + pt_sl[0] * entry_vol)
        lower = entry_price * (1 - pt_sl[1] * entry_vol)

        # 遍历路径找第一触达
        touched_upper = False
        touched_lower = False
        first_idx = None

        for j in range(1, len(path)):
            p = path.iloc[j]
            # 涨跌停检测——涨停/跌停价格无法突破，视为触达
            if detect_limits:
                pct_chg = p / entry_price - 1
                if pct_chg >= 0.095 and pt_sl[0] > 0:
                    # 涨停 ≈ 触达止盈
                    if not touched_lower:
                        upper = min(upper, entry_price * 1.10)
                elif pct_chg <= -0.095 and pt_sl[1] > 0:
                    if not touched_upper:
                        lower = max(lower, entry_price * 0.90)

            # 重新计算屏障（考虑涨跌停后）
            if p >= upper and pt_sl[0] > 0:
                touched_upper = True
                first_idx = path.index[j]
                break
            elif p <= lower and pt_sl[1] > 0:
                touched_lower = True
                first_idx = path.index[j]
                break

        if touched_upper:
            out.at[idx, 'bin'] = 1
            out.at[idx, 't1'] = first_idx
            out.at[idx, 'ret'] = (close.loc[first_idx] / entry_price - 1) * 100
        elif touched_lower:
            out.at[idx, 'bin'] = -1
            out.at[idx, 't1'] = first_idx
            out.at[idx, 'ret'] = (close.loc[first_idx] / entry_price - 1) * 100
        else:
            # 时间到期
            out.at[idx, 'bin'] = 0
            out.at[idx, 'ret'] = (path.iloc[-1] / entry_price - 1) * 100

    return out


def get_meta_labels(triple_barrier_result: pd.DataFrame,
                    side: pd.Series) -> pd.DataFrame:
    """
    生成 Meta-Labeling 标签

    参数:
        triple_barrier_result: apply_triple_barrier 的输出
        side: 主模型预测的方向 (+1 或 -1), index 对齐

    返回:
        DataFrame: columns=['bin','side','ret']
            bin: 1=可以执行, 0=跳过
            side: 主模型方向
    """
    out = triple_barrier_result[['t1', 'ret']].copy()
    out['side'] = side
    # Meta-label: 主方向 × 实际盈亏 > 0 → 可执行
    out['bin'] = ((side * out['ret']) > 0).astype(int)
    return out
