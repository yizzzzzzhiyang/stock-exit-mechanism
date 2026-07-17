"""
修复版：正确处理重叠入场信号，并放宽追损宽度测试
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from config import setup_proxy_bypass
setup_proxy_bypass()

from data.data_fetcher import fetch_data, eng_features
from labeling.cusum import cusum_filter_vol
from labeling.triple_barrier import apply_triple_barrier
from evaluation.metrics import (
    evaluate_strategy, compare_baselines, sharpe_ratio, max_drawdown,
    win_rate, profit_factor, compute_transaction_costs, get_cost_config
)


def trailing_stop_exit_v2(close, entries, vol, num_days=20,
                           lower_mult=1.0, trail_mult=2.0):
    """
    改进版退出 v2：正确处理重叠信号
    
    返回: 每个入场点的退出详情 DataFrame
    columns: entry_date, exit_date, exit_reason, entry_price, exit_price, ret
    """
    close_arr = close.values
    vol_arr = vol.values
    idx_arr = close.index.values
    
    trades = []
    
    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        if loc + 2 >= len(close_arr):
            continue
        
        entry_price = close_arr[loc]
        entry_vol = vol_arr[loc]
        if pd.isna(entry_vol) or entry_vol <= 0.001:
            entry_vol = 0.02
        
        lower = entry_price * (1 - lower_mult * entry_vol)
        highest = entry_price
        trail_stop = entry_price * (1 - trail_mult * entry_vol)
        
        end_loc = min(loc + num_days, len(close_arr) - 1)
        
        exit_reason = 'time'
        exit_idx = end_loc
        
        for i in range(loc + 1, end_loc + 1):
            p = close_arr[i]
            v = vol_arr[i]
            if pd.isna(v) or v <= 0.001:
                v = entry_vol
            
            if p > highest:
                highest = p
                new_stop = highest * (1 - trail_mult * v)
                if new_stop > trail_stop:
                    trail_stop = new_stop
            
            if p <= lower:
                exit_reason = 'stop_loss'
                exit_idx = i
                break
            elif p <= trail_stop:
                exit_reason = 'trailing'
                exit_idx = i
                break
        
        exit_price = close_arr[exit_idx]
        ret = (exit_price / entry_price - 1) * 100
        
        trades.append({
            'entry_date': entry,
            'exit_date': idx_arr[exit_idx],
            'exit_reason': exit_reason,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'ret': ret,
        })
    
    if not trades:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    
    # 生成信号：用持仓状态数组（0=空仓, 1=持仓）
    # 避免重叠覆盖问题：只要任意一个活跃交易在持有，信号就是1
    active = np.zeros(len(close_arr), dtype=bool)
    for t in trades:
        entry_loc = close.index.get_loc(t['entry_date'])
        exit_loc = close.index.get_loc(t['exit_date'])
        # 持有期: [entry_loc, exit_loc)，退出当天不算持有
        active[entry_loc:exit_loc] = True
    
    signals = pd.Series(active.astype(float), index=close.index)
    
    return trades_df, signals


def compute_trade_metrics(trades_df, close):
    """从交易记录计算策略指标"""
    if len(trades_df) == 0:
        return {'name': 'empty', 'sharpe': 0, 'win_rate': 0, 'profit_factor': 0, 
                'max_dd': 0, 'ann_return': 0, 'n_trades': 0}
    
    # 用信号方式计算
    _, signals = trailing_stop_exit_v2.__wrapped__ = None  # placeholder
    # 直接用 trades 构造信号
    close_arr = close.values
    idx_arr = close.index.values
    active = np.zeros(len(close_arr), dtype=bool)
    for _, t in trades_df.iterrows():
        entry_loc = close.index.get_loc(t['entry_date'])
        exit_loc = close.index.get_loc(t['exit_date'])
        active[entry_loc:exit_loc] = True
    
    signals = pd.Series(active.astype(float), index=close.index)
    return evaluate_strategy(signals, close, 'trailing', num_trials=4)


def run_comparison(ts_code, start, end, cusum_mult=1.5, num_days=20):
    """对比 TB vs 追损(多宽度)"""
    print(f'\n{"="*70}')
    print(f'  {ts_code}')
    print(f'{"="*70}')
    
    df = fetch_data(ts_code, start, end)
    df = eng_features(df).dropna()
    close = df['close']
    vol = df['v20'].clip(lower=0.001)
    atr = df['atr']
    
    t_events = cusum_filter_vol(close, vol, cusum_mult)
    if len(t_events) < 10:
        t_events = cusum_filter_vol(close, vol, cusum_mult * 0.5)
    
    print(f'  数据: {len(df)}日  CUSUM事件: {len(t_events)}')
    
    # 1) TB[2,1] — 用 apply_triple_barrier
    labels_tb = apply_triple_barrier(close, t_events, [2, 1], vol, num_days)
    
    # 构造 TB 信号
    active = np.zeros(len(close), dtype=bool)
    for entry in t_events:
        if entry in labels_tb.index:
            end_dt = labels_tb.loc[entry, 't1']
            if isinstance(end_dt, pd.Timestamp):
                el = close.index.get_loc(entry)
                xl = close.index.get_loc(end_dt)
                active[el:xl] = True
    signals_tb = pd.Series(active.astype(float), index=close.index)
    
    m_tb = evaluate_strategy(signals_tb, close, 'TB[2,1]', num_trials=4)
    
    # 统计 TB 交易明细
    tb_details = labels_tb.copy()
    tb_details['ret'] = tb_details['ret'].fillna(0)
    n_up = (tb_details['bin'] == 1).sum()
    n_dn = (tb_details['bin'] == -1).sum()
    n_tm = (tb_details['bin'] == 0).sum()
    avg_win = tb_details[tb_details['bin']==1]['ret'].mean() if n_up > 0 else 0
    avg_loss = tb_details[tb_details['bin']==-1]['ret'].mean() if n_dn > 0 else 0
    
    print(f'  TB[2,1]: 止盈{n_up} 止损{n_dn} 超时{n_tm}  均盈{avg_win:.1f}% 均亏{avg_loss:.1f}%')
    
    # 2) 追损多宽度
    print(f'\n  {"策略":<16} {"Sharpe":>7} {"胜率":>7} {"盈亏比":>7} {"回撤":>7} {"年化":>8} {"均盈%":>7} {"均亏%":>7} {"交易"}')
    print(f'  {"-"*78}')
    
    results = [m_tb]
    
    for trail_m in [1.5, 2.0, 2.5, 3.0]:
        trades_df, signals = trailing_stop_exit_v2(
            close, t_events, vol, num_days, lower_mult=1.0, trail_mult=trail_m
        )
        if len(trades_df) == 0:
            continue
        
        m = evaluate_strategy(signals, close, f'追损{trail_m}σ', num_trials=4)
        results.append(m)
        
        # 交易明细
        wins = trades_df[trades_df['exit_reason']=='trailing']
        losses = trades_df[trades_df['exit_reason']=='stop_loss']
        times = trades_df[trades_df['exit_reason']=='time']
        
        avg_w = wins['ret'].mean() if len(wins) > 0 else 0
        avg_l = losses['ret'].mean() if len(losses) > 0 else 0
        
        print(f'  {m["name"]:<16} {m["sharpe"]:>7.3f} {m["win_rate"]:>7.1%} '
              f'{m["profit_factor"]:>7.2f} {m["max_dd"]:>7.1%} {m["ann_return"]:>8.1%} '
              f'{avg_w:>7.1f} {avg_l:>7.1f} '
              f'追{len(wins)}/止{len(losses)}/时{len(times)}')
    
    # 3) 基准
    baselines = compare_baselines(close, t_events, atr)
    
    print(f'\n  {"-"*78}')
    print(f'  基准策略:')
    for _, r in baselines.iterrows():
        if r['name'] != '买入持有':
            print(f'    {r["name"]:<20} SR={r["sharpe"]:.3f}  WR={r["win_rate"]:.1%}  '
                  f'MDD={r["max_dd"]:.1%}')
    
    # 最优
    best = max(results, key=lambda x: x['sharpe'])
    print(f'\n  → 最优: {best["name"]} (SR={best["sharpe"]:.3f})')
    
    return results


if __name__ == '__main__':
    stocks = [
        ('002714.SZ', '牧原股份'),
        ('601318.SH', '中国平安'),
        ('000858.SZ', '五粮液'),
        ('688336.SH', '三生国健'),
        ('600875.SH', '东方电气'),
    ]
    
    # 先测试牧原一只，确认无 bug
    results = run_comparison('002714.SZ', '2015-01-01', '2026-07-03')
