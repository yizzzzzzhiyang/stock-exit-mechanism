"""
修复版 v2：全部5只股票对比
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
from evaluation.metrics import evaluate_strategy, compare_baselines


def trailing_stop_exit_v2(close, entries, vol, num_days=20,
                           lower_mult=1.0, trail_mult=2.0):
    close_arr = close.values
    vol_arr = vol.values
    idx_arr = close.index.values
    
    trades = []
    for entry in entries:
        if entry not in close.index: continue
        loc = close.index.get_loc(entry)
        if loc + 2 >= len(close_arr): continue
        
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
            if pd.isna(v) or v <= 0.001: v = entry_vol
            
            if p > highest:
                highest = p
                new_stop = highest * (1 - trail_mult * v)
                if new_stop > trail_stop: trail_stop = new_stop
            
            if p <= lower:
                exit_reason = 'stop_loss'; exit_idx = i; break
            elif p <= trail_stop:
                exit_reason = 'trailing'; exit_idx = i; break
        
        trades.append({
            'entry_date': entry, 'exit_date': idx_arr[exit_idx],
            'exit_reason': exit_reason, 'ret': (close_arr[exit_idx]/entry_price-1)*100
        })
    
    if not trades: return pd.DataFrame(), None
    
    trades_df = pd.DataFrame(trades)
    active = np.zeros(len(close_arr), dtype=bool)
    for _, t in trades_df.iterrows():
        el = close.index.get_loc(t['entry_date'])
        xl = close.index.get_loc(t['exit_date'])
        active[el:xl] = True
    signals = pd.Series(active.astype(float), index=close.index)
    return trades_df, signals


def run_one(ts_code, name, start, end):
    print(f'\n{"="*65}')
    print(f'  {name} ({ts_code})')
    print(f'{"="*65}')
    
    df = fetch_data(ts_code, start, end)
    df = eng_features(df).dropna()
    close = df['close']; vol = df['v20'].clip(lower=0.001); atr = df['atr']
    
    t_events = cusum_filter_vol(close, vol, 1.5)
    if len(t_events) < 10:
        t_events = cusum_filter_vol(close, vol, 0.75)
    
    # TB[2,1]
    labels_tb = apply_triple_barrier(close, t_events, [2, 1], vol, 20)
    active = np.zeros(len(close), dtype=bool)
    for entry in t_events:
        if entry in labels_tb.index:
            ed = labels_tb.loc[entry, 't1']
            if isinstance(ed, pd.Timestamp):
                active[close.index.get_loc(entry):close.index.get_loc(ed)] = True
    signals_tb = pd.Series(active.astype(float), index=close.index)
    m_tb = evaluate_strategy(signals_tb, close, 'TB[2,1]', num_trials=4)
    
    n_up = (labels_tb['bin']==1).sum(); n_dn = (labels_tb['bin']==-1).sum()
    avg_w = labels_tb[labels_tb['bin']==1]['ret'].mean()
    avg_l = labels_tb[labels_tb['bin']==-1]['ret'].mean()
    
    # 追损
    summary = [m_tb]
    print(f'  {"策略":<14} {"Sharpe":>7} {"胜率":>7} {"盈亏比":>7} {"回撤":>7} {"年化":>8}  {"追/止/时"}')
    print(f'  {"-"*60}')
    print(f'  {"TB[2,1]":<14} {m_tb["sharpe"]:>7.3f} {m_tb["win_rate"]:>7.1%} '
          f'{m_tb["profit_factor"]:>7.2f} {m_tb["max_dd"]:>7.1%} {m_tb["ann_return"]:>8.1%}  '
          f'止盈{n_up}/止损{n_dn}')
    
    for trail_m in [1.5, 2.0, 2.5, 3.0]:
        trades_df, signals = trailing_stop_exit_v2(close, t_events, vol, 20, 1.0, trail_m)
        if len(trades_df)==0: continue
        m = evaluate_strategy(signals, close, f'追损{trail_m}σ', num_trials=4)
        summary.append(m)
        n_tr = (trades_df['exit_reason']=='trailing').sum()
        n_sl = (trades_df['exit_reason']=='stop_loss').sum()
        n_tm = (trades_df['exit_reason']=='time').sum()
        print(f'  {m["name"]:<14} {m["sharpe"]:>7.3f} {m["win_rate"]:>7.1%} '
              f'{m["profit_factor"]:>7.2f} {m["max_dd"]:>7.1%} {m["ann_return"]:>8.1%}  '
              f'追{n_tr}/止{n_sl}/时{n_tm}')
    
    best = max(summary, key=lambda x: x['sharpe'])
    print(f'  {"-"*60}')
    print(f'  → {best["name"]} SR={best["sharpe"]:.3f}')
    
    return {r['name']: r['sharpe'] for r in summary}


if __name__ == '__main__':
    stocks = [
        ('002714.SZ', '牧原股份'),
        ('601318.SH', '中国平安'),
        ('000858.SZ', '五粮液'),
        ('688336.SH', '三生国健'),
        ('600875.SH', '东方电气'),
    ]
    
    all_results = {}
    for code, name in stocks:
        try:
            all_results[name] = run_one(code, name, '2015-01-01', '2026-07-03')
        except Exception as e:
            print(f'  ❌ {name}: {e}')
    
    print(f'\n\n{"="*80}')
    print(f'  汇总: 各股票最优 Sharpe')
    print(f'{"="*80}')
    print(f'  {"股票":<10} {"TB[2,1]":>8} {"追损1.5σ":>9} {"追损2.0σ":>9} {"追损2.5σ":>9} {"追损3.0σ":>9}  {"最优":>10}')
    print(f'  {"-"*65}')
    for name, r in all_results.items():
        vals = [f'{r.get(k, 0):.3f}' for k in ['TB[2,1]', '追损1.5σ', '追损2.0σ', '追损2.5σ', '追损3.0σ']]
        best_k = max(r, key=r.get)
        print(f'  {name:<10} {vals[0]:>8} {vals[1]:>9} {vals[2]:>9} {vals[3]:>9} {vals[4]:>9}  {best_k}({r[best_k]:.3f})')
