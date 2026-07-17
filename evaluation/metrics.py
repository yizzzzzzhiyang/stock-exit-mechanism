"""
评估指标 + 基准对比 + Deflated Sharpe Ratio + 交易成本

v2.0: 加入交易成本（佣金 + 印花税 + 滑点），使回测更接近实盘。
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


# ────────────── 交易成本配置 ──────────────
# 可通过环境变量覆盖
import os
from typing import Optional

def get_cost_config() -> dict:
    """
    返回交易成本配置（单位：比例）
    
    环境变量:
      COST_COMMISSION  — 单边佣金率 (默认 0.00025 = 万2.5)
      COST_STAMP_TAX   — 印花税率 (默认 0.001 = 千1, 仅卖出)
      COST_SLIPPAGE    — 滑点率 (默认 0.001 = 0.1%)
    """
    return {
        'commission': float(os.environ.get('COST_COMMISSION', 0.00025)),
        'stamp_tax':  float(os.environ.get('COST_STAMP_TAX', 0.001)),
        'slippage':   float(os.environ.get('COST_SLIPPAGE', 0.001)),
    }


def compute_transaction_costs(signals: pd.Series, prices: pd.Series,
                              cost_config: Optional[dict] = None) -> tuple:
    """
    计算交易成本扣除后的收益率序列

    成本触发规则:
      信号变化 → 发生交易 → 扣除对应成本
      - 0 → ±x:     买入，扣 commission + slippage（按仓位比例）
      - ±x → 0:      卖出，扣 commission + stamp_tax + slippage
      - ±x → ±y:     调仓（同方向），扣 |y-x| × (commission + slippage)
      - +x → -y:     先卖后买，全额费用
      - 信号不变:      无交易，不扣费

    支持连续仓位信号 [-1, 1]，成本按仓位比例计算。
    """
    if cost_config is None:
        cost_config = get_cost_config()

    commission = cost_config['commission']
    stamp_tax = cost_config['stamp_tax']
    slippage = cost_config['slippage']

    buy_cost = commission + slippage          # 买入单边
    sell_cost = commission + stamp_tax + slippage  # 卖出单边

    common_idx = signals.dropna().index.intersection(prices.index)
    s = signals.loc[common_idx]
    p = prices.loc[common_idx]

    ret = p.pct_change().loc[common_idx[1]:]
    prev_s = s.shift(1).loc[common_idx[1]:]
    s_current = s.loc[common_idx[1]:]

    raw_ret = prev_s * ret
    costs = pd.Series(0.0, index=raw_ret.index)
    trade_count = 0

    for i in range(len(s_current)):
        prev = prev_s.iloc[i]
        curr = s_current.iloc[i]

        if prev == 0 and curr != 0:
            # 入场
            costs.iloc[i] = abs(curr) * buy_cost
            trade_count += 1
        elif prev != 0 and curr == 0:
            # 离场
            costs.iloc[i] = abs(prev) * sell_cost
            trade_count += 1
        elif prev != 0 and curr != 0 and np.sign(prev) != np.sign(curr):
            # 反向: 先卖后买
            costs.iloc[i] = abs(prev) * sell_cost + abs(curr) * buy_cost
            trade_count += 2
        elif prev != 0 and curr != 0 and abs(curr - prev) > 1e-8:
            # 同向调仓
            costs.iloc[i] = abs(curr - prev) * buy_cost
            trade_count += 1

    net_ret = raw_ret - costs
    return net_ret, costs, trade_count


# ────────────── 基本指标 ──────────────


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0,
                 periods: int = 252) -> float:
    """年化夏普比率"""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = returns - rf / periods
    return np.sqrt(periods) * np.mean(excess) / np.std(returns)


def sortino_ratio(returns: np.ndarray, rf: float = 0.0,
                  periods: int = 252) -> float:
    """索提诺比率（只用下行波动率）"""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    downside = np.std(returns[returns < 0])
    if downside == 0:
        return 0.0 if np.mean(excess) >= 0 else -10.0
    return np.sqrt(periods) * np.mean(excess) / downside


def max_drawdown(equity: np.ndarray) -> float:
    """最大回撤"""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return np.min(dd)


def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """卡玛比率：年化收益 / 最大回撤绝对值"""
    if len(returns) < 2:
        return 0.0
    ann_ret = np.mean(returns) * periods
    eq = np.cumprod(1 + returns)
    mdd = max_drawdown(eq)
    if abs(mdd) < 1e-10:
        return 0.0
    return ann_ret / abs(mdd)


def win_rate(returns: np.ndarray) -> float:
    """胜率"""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    """盈亏比"""
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def deflated_sharpe_ratio(observed_sharpe: float,
                          num_trials: int,
                          max_theoretical_sharpe: float = 1.0,
                          skew: float = 0.0,
                          kurt: float = 3.0,
                          t: int = 252) -> float:
    """
    Deflated Sharpe Ratio (DSR)
    修正多重假设检验偏差 — López de Prado & Lewis (2018)

    参数:
        observed_sharpe: 观察到的夏普比率
        num_trials: 试过的策略/参数组合数量
        max_theoretical_sharpe: 理论上可能的最高夏普
        skew: 收益率偏度
        kurt: 收益率峰度
        t: 样本数

    返回:
        DSR: 经多重假设检验修正后的夏普
    """
    if num_trials <= 1:
        return observed_sharpe

    # Estér 分布下的夏普标准差
    var_sharpe = (1 + 0.5 * skew * observed_sharpe -
                  0.25 * (kurt - 3) * observed_sharpe ** 2) / (t - 1)

    if var_sharpe <= 0:
        return observed_sharpe

    std_sharpe = np.sqrt(var_sharpe)

    # 多重比较下的基准夏普（E[min(SR)] 的近似）
    e_min = norm.ppf(1 - 1 / num_trials) * std_sharpe + max_theoretical_sharpe

    # DSR
    dsr = (observed_sharpe - e_min) / std_sharpe
    return dsr


def evaluate_strategy(signals: pd.Series, prices: pd.Series,
                      name: str = 'Strategy',
                      num_trials: int = 1,
                      include_costs: bool = True,
                      cost_config: Optional[dict] = None) -> dict:
    """
    评估策略表现

    参数:
        signals: 信号序列 {-1, 0, 1}, index=日期
        prices: 价格序列, index=日期
        name: 策略名称
        num_trials: 用于 DSR 的 trial 数
        include_costs: 是否扣除交易成本（默认 True）
        cost_config: 成本配置字典，None 则用默认

    返回:
        metrics dict
    """
    # 对齐
    common_idx = signals.dropna().index.intersection(prices.index)
    s = signals.loc[common_idx]
    p = prices.loc[common_idx]

    # 前一日信号
    prev_s = s.shift(1).loc[common_idx[1]:]
    ret = p.pct_change().loc[common_idx[1]:]

    # 原始策略收益
    strategy_ret = prev_s * ret
    strategy_ret = strategy_ret.dropna()

    if len(strategy_ret) < 5:
        return {'name': name, 'error': 'insufficient data'}

    # 交易成本扣除
    trade_count = 0
    total_cost = 0.0
    if include_costs:
        net_ret, cost_series, trade_count = compute_transaction_costs(
            signals, prices, cost_config
        )
        total_cost = float(cost_series.sum())
    else:
        net_ret = strategy_ret

    # 只考虑有交易的日期
    trade_days = net_ret[net_ret != 0]
    eq = np.cumprod(1 + net_ret.values)

    sr = sharpe_ratio(net_ret.values)
    dsr = deflated_sharpe_ratio(sr, num_trials)
    sortino = sortino_ratio(net_ret.values)
    mdd = max_drawdown(eq)
    calmar = calmar_ratio(net_ret.values)
    wr = win_rate(trade_days.values) if len(trade_days) > 0 else 0
    pf = profit_factor(trade_days.values) if len(trade_days) > 0 else 0
    ann_ret = np.mean(net_ret.values) * 252
    ann_vol = np.std(net_ret.values) * np.sqrt(252)

    metrics = {
        'name': name,
        'sharpe': round(sr, 3),
        'dsr': round(dsr, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(mdd, 4),
        'calmar': round(calmar, 3),
        'win_rate': round(wr, 3),
        'profit_factor': round(pf, 3),
        'total_return': round(eq[-1] - 1, 4),
        'ann_return': round(ann_ret, 4),
        'ann_vol': round(ann_vol, 4),
        'n_trades': trade_count if include_costs else int((s.shift(1) != 0).sum()),
        'n_active_days': len(net_ret),
        'total_cost': round(total_cost, 6),
    }
    return metrics


# ────────────── 基准策略 ──────────────


def fixed_stop_loss(entries: pd.DatetimeIndex, prices: pd.Series,
                    stop_loss: float = 0.05, take_profit: float = 0.10,
                    max_hold: int = 60) -> pd.Series:
    """
    固定止损止盈基准

    参数:
        entries: 入场时间
        prices: 价格序列
        stop_loss: 止损比例 (例如 0.05 = 5%)
        take_profit: 止盈比例
        max_hold: 最长持仓

    返回:
        signals: {-1, 0, 1} 信号序列
    """
    signals = pd.Series(0, index=prices.index)
    for entry in entries:
        if entry not in prices.index:
            continue
        masked = prices.index[prices.index >= entry]
        if len(masked) < 2:
            continue
        end = masked[min(max_hold, len(masked) - 1)]
        path = prices.loc[entry:end]
        entry_p = path.iloc[0]

        exited = False
        for t in path.index[1:]:
            ret = path[t] / entry_p - 1
            if ret <= -stop_loss:
                signals[t] = -1  # 止损退出
                exited = True
                break
            elif ret >= take_profit:
                signals[t] = 1  # 止盈退出
                exited = True
                break
        if not exited:
            signals[end] = 1  # 时间退出

    return signals


def atr_trailing_stop(entries: pd.DatetimeIndex, prices: pd.Series,
                      atr: pd.Series, atr_mult: float = 2.0,
                      max_hold: int = 60) -> pd.Series:
    """
    ATR 跟踪止损基准
    """
    signals = pd.Series(0, index=prices.index)
    close_arr = prices.values
    atr_arr = atr.values
    idx_arr = prices.index.values

    for entry in entries:
        if entry not in prices.index:
            continue
        loc = prices.index.get_loc(entry)
        if loc + max_hold >= len(prices):
            continue

        end = min(loc + max_hold, len(prices) - 1)
        highest = close_arr[loc]
        stop = highest - atr_mult * atr_arr[loc]

        for i in range(loc + 1, end):
            p = close_arr[i]
            if p > highest:
                highest = p
                stop = highest - atr_mult * atr_arr[i]
            if p < stop:
                signals[idx_arr[i]] = -1
                break
        else:
            signals[idx_arr[end]] = 1  # 时间退出

    return signals


def compare_baselines(close: pd.Series, entries: pd.DatetimeIndex,
                      atr: pd.Series = None,
                      include_costs: bool = True) -> pd.DataFrame:
    """
    对比所有基准策略

    返回:
        DataFrame: rows=策略, cols=指标
    """
    results = []
    strategies = [
        ('固定止损 5%', lambda: fixed_stop_loss(entries, close, 0.05, 0.10)),
        ('固定止损 3%', lambda: fixed_stop_loss(entries, close, 0.03, 0.06)),
        ('固定止损 8%', lambda: fixed_stop_loss(entries, close, 0.08, 0.15)),
    ]
    if atr is not None:
        strategies += [
            ('ATR×2 跟踪', lambda: atr_trailing_stop(entries, close, atr, 2.0)),
            ('ATR×3 跟踪', lambda: atr_trailing_stop(entries, close, atr, 3.0)),
        ]
    strategies += [
        ('买入持有', lambda: pd.Series(1, index=close.index)),
    ]

    for name, fn in strategies:
        signals = fn()
        if isinstance(signals, pd.Series) and len(signals) > 0:
            metrics = evaluate_strategy(
                signals, close, name, include_costs=include_costs
            )
            results.append(metrics)

    return pd.DataFrame(results)
