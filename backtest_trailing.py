"""
改进版退出机制回测：下屏障 + 追损线 + 时间屏障
对比纯 TB、固定止损、不同追损宽度的效果
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
    evaluate_strategy, compare_baselines, fixed_stop_loss, atr_trailing_stop
)


def trailing_stop_exit(close, entries, vol, num_days=20,
                        lower_mult=1.0, trail_mult=1.5):
    """
    改进版退出：下屏障（固定）+ 追损线（动态）+ 时间屏障
    
    参数:
        close:   价格序列
        entries: 入场时间戳 (CUSUM 事件)
        vol:     波动率序列 σ
        num_days: 最大持仓天数
        lower_mult: 下屏障倍数
        trail_mult: 追损线倍数（从最高点回撤 trail_mult * σ 触发）
    
    返回: signals Series (1=持有, 0=空仓)
    """
    signals = pd.Series(0, index=close.index, dtype=float)
    close_arr = close.values
    vol_arr = vol.values
    idx_arr = close.index.values
    
    for entry in entries:
        if entry not in close.index:
            continue
        loc = close.index.get_loc(entry)
        if loc + 2 >= len(close):
            continue
        
        entry_price = close_arr[loc]
        entry_vol = vol_arr[loc]
        if pd.isna(entry_vol) or entry_vol <= 0:
            entry_vol = 0.02
        
        # 下屏障：固定，以入场价为基准
        lower = entry_price * (1 - lower_mult * entry_vol)
        
        # 追损线初始值：入场时也设一个初始止损
        highest = entry_price
        trail_stop = highest * (1 - trail_mult * entry_vol)
        
        end_loc = min(loc + num_days, len(close_arr) - 1)
        
        for i in range(loc + 1, end_loc + 1):
            p = close_arr[i]
            v = vol_arr[i]
            if pd.isna(v) or v <= 0:
                v = entry_vol
            
            # 更新最高价和追损线（只上不下）
            if p > highest:
                highest = p
                trail_stop = max(trail_stop, highest * (1 - trail_mult * v))
            
            # 检查触发
            if p <= lower:
                # 触发下屏障（止损），标记卖出日
                signals[idx_arr[i]] = 0
                # 入场日到卖出日前一天持有
                signals.loc[entry:idx_arr[i-1]] = 1
                break
            elif p <= trail_stop:
                # 触发追损线（止盈）
                signals[idx_arr[i]] = 0
                signals.loc[entry:idx_arr[i-1]] = 1
                break
        else:
            # 时间到期
            signals.loc[entry:idx_arr[end_loc]] = 1
    
    return signals


def run_single_stock(ts_code, start, end, cusum_mult=1.5, num_days=20):
    """对单只股票运行全部策略对比"""
    print(f'\n{"="*70}')
    print(f'  {ts_code}')
    print(f'{"="*70}')
    
    # 获取数据
    df = fetch_data(ts_code, start, end)
    df = eng_features(df).dropna()
    close = df['close']
    vol = df['v20'].clip(lower=0.001)
    atr = df['atr']
    
    # CUSUM 事件
    t_events = cusum_filter_vol(close, vol, cusum_mult)
    if len(t_events) < 10:
        t_events = cusum_filter_vol(close, vol, cusum_mult * 0.5)
    
    n_events = len(t_events)
    
    # ── 策略1: 纯 TB（上下屏障固定，无追损）──
    labels_tb = apply_triple_barrier(close, t_events, [2, 1], vol, num_days)
    signals_tb = pd.Series(0, index=close.index, dtype=float)
    for entry in t_events:
        if entry in labels_tb.index:
            row = labels_tb.loc[entry]
            end_dt = row['t1']
            if isinstance(end_dt, pd.Timestamp):
                signals_tb.loc[entry:end_dt] = 1
    metrics_tb = evaluate_strategy(signals_tb, close, 'TB[2,1]', num_trials=4)
    
    # ── 策略2-4: 改进版追损 (不同宽度) ──
    results = [metrics_tb]
    
    for trail_m in [1.5, 2.0, 2.5]:
        signals = trailing_stop_exit(close, t_events, vol, num_days,
                                     lower_mult=1.0, trail_mult=trail_m)
        m = evaluate_strategy(signals, close,
                              f'追损{trail_m}σ', num_trials=4)
        results.append(m)
    
    # ── 基准策略 ──
    baselines = compare_baselines(close, t_events, atr)
    
    # ── 输出 ──
    print(f'  事件数: {n_events}  数据: {len(df)}日  {start}~{end}')
    print(f'  {"策略":<14} {"Sharpe":>7} {"胜率":>7} {"盈亏比":>7} {"回撤":>7} {"年化":>8}')
    print(f'  {"-"*50}')
    
    all_results = results + [
        {'name': r['name'], 'sharpe': r['sharpe'], 'win_rate': r['win_rate'],
         'profit_factor': r['profit_factor'], 'max_dd': r['max_dd'],
         'ann_return': r['ann_return']}
        for _, r in baselines.iterrows() if r['name'] != '买入持有'
    ]
    
    best_sr = -999
    best_name = ''
    for r in all_results:
        print(f'  {r["name"]:<14} {r["sharpe"]:>7.3f} {r["win_rate"]:>7.1%} '
              f'{r["profit_factor"]:>7.2f} {r["max_dd"]:>7.1%} {r["ann_return"]:>8.1%}')
        if r['sharpe'] > best_sr:
            best_sr = r['sharpe']
            best_name = r['name']
    
    print(f'  {"-"*50}')
    print(f'  → 最优: {best_name} (Sharpe={best_sr:.3f})')
    
    return {
        'ts_code': ts_code,
        'n_events': n_events,
        'results': all_results,
        'best': best_name,
        'best_sr': best_sr
    }


if __name__ == '__main__':
    stocks = [
        ('002714.SZ', '牧原股份'),
        ('601318.SH', '中国平安'),
        ('000858.SZ', '五粮液'),
        ('688336.SH', '三生国健'),
        ('600875.SH', '东方电气'),
    ]
    
    start = '2015-01-01'
    end = '2026-07-03'
    
    all_summary = []
    
    for code, name in stocks:
        try:
            result = run_single_stock(code, start, end)
            all_summary.append(result)
        except Exception as e:
            print(f'\n  {name}({code}): ❌ 出错 — {e}')
    
    # ── 汇总 ──
    print(f'\n\n{"="*70}')
    print(f'  汇总对比')
    print(f'{"="*70}')
    print(f'  {"股票":<12} {"事件":>5} {"TB[2,1]":>8} {"追损1.5σ":>9} {"追损2.0σ":>9} {"追损2.5σ":>9}  {"最优"}')
    print(f'  {"-"*70}')
    
    for r in all_summary:
        res = {x['name']: x['sharpe'] for x in r['results']}
        print(f'  {r["ts_code"]:<12} {r["n_events"]:>5} '
              f'{res.get("TB[2,1]", 0):>8.3f} '
              f'{res.get("追损1.5σ", 0):>9.3f} '
              f'{res.get("追损2.0σ", 0):>9.3f} '
              f'{res.get("追损2.5σ", 0):>9.3f}  '
              f'{r["best"]} ({r["best_sr"]:.3f})')
