"""
上涨通道退出策略库

实现 5 种上涨趋势退出策略 + 基准对比:
  1. 均线移动止盈 (MA Trailing Stop)
  2. 最高价回撤百分比 (Drawdown from Peak)
  3. ATR 动态跟踪止盈 (ATR Trailing Stop)
  4. 通道上轨分批止盈 (Channel Partial Exit)
  5. 抛物线 SAR 跟踪止盈 (Parabolic SAR)
"""
import numpy as np
import pandas as pd


def ma_trailing_exit(close: pd.Series, entries: pd.DatetimeIndex,
                     ma_period: int = 20) -> pd.Series:
    """
    策略 1: 均线移动止盈
    持有期间收盘价跌破 MA 则退出，否则持有到期末
    """
    ma = close.rolling(ma_period).mean()
    signals = pd.Series(0, index=close.index)

    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        for i in range(loc + 1, len(close)):
            if close.iloc[i] < ma.iloc[i]:
                signals.loc[entry:close.index[i]] = 1
                break
            if i == len(close) - 1:
                signals.loc[entry:close.index[i]] = 1

    return signals


def drawdown_exit(close: pd.Series, entries: pd.DatetimeIndex,
                  drawdown_pct: float = 0.08, max_hold: int = 120) -> pd.Series:
    """
    策略 2: 最高价回撤百分比止盈
    从入场后最高价回撤超过 drawdown_pct 则退出
    """
    signals = pd.Series(0, index=close.index)

    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        end = min(loc + max_hold + 1, len(close))
        highest = close.iloc[loc]

        for i in range(loc + 1, end):
            p = close.iloc[i]
            if p > highest:
                highest = p
            if p < highest * (1 - drawdown_pct):
                signals.loc[entry:close.index[i]] = 1
                break
        else:
            signals.loc[entry:close.index[end-1]] = 1

    return signals


def atr_trailing_exit(close: pd.Series, atr: pd.Series,
                      entries: pd.DatetimeIndex, atr_mult: float = 3.0,
                      max_hold: int = 120) -> pd.Series:
    """
    策略 3: ATR 动态跟踪止盈
    止盈位 = 入场以来最高价 - N × ATR
    """
    signals = pd.Series(0, index=close.index)
    c = close.values; a = atr.values; idx = close.index

    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        end = min(loc + max_hold + 1, len(close))
        highest = c[loc]
        stop = highest - atr_mult * a[loc]

        for i in range(loc + 1, end):
            if c[i] > highest:
                highest = c[i]
                stop = highest - atr_mult * a[i]
            if c[i] < stop:
                signals.loc[idx[loc]:idx[i]] = 1
                break
        else:
            signals.loc[idx[loc]:idx[end-1]] = 1

    return signals


def channel_partial_exit(close: pd.Series, entries: pd.DatetimeIndex,
                         bb_period: int = 20, bb_std: float = 2.0,
                         sell_ratio: float = 0.5, trail_ma: int = 20,
                         max_hold: int = 120) -> pd.Series:
    """
    策略 4: 通道上轨分批止盈
    触布林上轨 → 卖 sell_ratio 比例
    剩余仓位用均线移动止盈
    返回加权信号（0.5 = 半仓）
    """
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    upper = mid + bb_std * std
    ma_trail = close.rolling(trail_ma).mean()

    signals = pd.Series(0.0, index=close.index)

    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        end = min(loc + max_hold + 1, len(close))
        partial_taken = False

        for i in range(loc + 1, end):
            p = close.iloc[i]

            # 触上轨 → 分批止盈
            if not partial_taken and p >= upper.iloc[i]:
                signals.loc[entry:close.index[i]] = sell_ratio
                partial_taken = True
                # 剩余仓位从下一根 K 线开始跟踪
                remaining_entry = i + 1 if i + 1 < end else i
                # 剩余仓位用均线跟踪
                for j in range(remaining_entry, end):
                    if close.iloc[j] < ma_trail.iloc[j]:
                        signals.loc[close.index[i+1]:close.index[j]] += (1 - sell_ratio)
                        break
                    if j == end - 1:
                        signals.loc[close.index[i+1]:close.index[j]] += (1 - sell_ratio)
                break

            # 跌破均线（没到过上轨）
            if p < ma_trail.iloc[i]:
                signals.loc[entry:close.index[i]] = 1
                break
        else:
            if not partial_taken:
                signals.loc[entry:close.index[end-1]] = 1

    # clip to [-1, 1]
    signals = signals.clip(-1, 1)
    return signals


def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.20) -> pd.Series:
    """
    策略 5: 抛物线 SAR
    返回 SAR 值序列（用于止盈判断）

    SAR 只在上涨趋势中计算（假设入场后为多头）。
    当价格跌破 SAR 时退出。
    """
    n = len(high)
    sar = np.full(n, np.nan)
    ep = np.full(n, np.nan)   # extreme point
    af = np.full(n, af_start)
    trend_up = np.ones(n, dtype=bool)  # 始终假设多头

    # 初始化：前两根 K 线的低点
    sar[0] = low.iloc[0]
    ep[0] = high.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i-1]
        prev_ep = ep[i-1]

        if trend_up[i-1]:
            # 多头：SAR 上移
            sar[i] = prev_sar + af[i-1] * (prev_ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i-1])  # 不能高于前低
            if i > 1:
                sar[i] = min(sar[i], low.iloc[i-2])

            if high.iloc[i] > prev_ep:
                ep[i] = high.iloc[i]
                af[i] = min(af[i-1] + af_step, af_max)
            else:
                ep[i] = prev_ep
                af[i] = af[i-1]

            # 反转检查
            if low.iloc[i] < sar[i]:
                trend_up[i] = False
                sar[i] = prev_ep
                ep[i] = low.iloc[i]
                af[i] = af_start
            else:
                trend_up[i] = True

    return pd.Series(sar, index=high.index)


def sar_trailing_exit(close: pd.Series, high: pd.Series, low: pd.Series,
                      entries: pd.DatetimeIndex, max_hold: int = 120) -> pd.Series:
    """
    策略 5: 抛物线 SAR 跟踪止盈
    从入场点开始计算 SAR，价格跌破 SAR 则退出
    """
    # 为每笔交易计算 SAR
    signals = pd.Series(0, index=close.index)

    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        end = min(loc + max_hold + 1, len(close))

        # 从入场日开始计算 SAR
        seg_high = high.iloc[loc:end]
        seg_low = low.iloc[loc:end]
        sar_vals = parabolic_sar(seg_high, seg_low)

        entry_price = close.iloc[loc]
        for i_offset, sar_val in enumerate(sar_vals):
            if pd.isna(sar_val):
                continue
            actual_idx = loc + i_offset
            if actual_idx >= len(close):
                break
            if close.iloc[actual_idx] < sar_val and actual_idx > loc:
                signals.loc[entry:close.index[actual_idx]] = 1
                break
        else:
            signals.loc[entry:close.index[end-1]] = 1

    return signals
