"""
A 股数据获取 + 特征工程
绕过代理直连 tushare
"""
import os
os.environ['no_proxy'] = 'tushare.pro,api.tushare.pro,*'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tushare as ts


def _clean_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """清洗指数数据，统一列名"""
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    # 指数只有 open/high/low/close/vol
    rename = {}
    if 'vol' in df.columns:
        rename['vol'] = 'volume'
    df.rename(columns=rename, inplace=True)
    df.index.name = 'date'
    return df


def fetch_data(ts_code='000001.SZ', start_date='2008-01-01',
               end_date='2026-07-01', adj='hfq') -> pd.DataFrame:
    """获取 A 股日线（后复权）
    支持: 个股代码(如 000001.SZ), 指数代码(如 000300.SH)
    """
    pro = ts.pro_api()
    # 尝试用 daily 接口（适用于个股）
    df = pro.daily(ts_code=ts_code,
                   start_date=start_date.replace('-', ''),
                   end_date=end_date.replace('-', ''),
                   adj=adj)
    if df is None or len(df) == 0:
        # 可能是指数，用 index_daily 接口
        df = pro.index_daily(ts_code=ts_code,
                             start_date=start_date.replace('-', ''),
                             end_date=end_date.replace('-', ''))
        if df is not None and len(df) > 0:
            # 指数没有复权概念，直接返回
            return _clean_index_df(df)
        raise ValueError(f'tushare 空数据: {ts_code}')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    df.rename(columns={'vol': 'volume'}, inplace=True)
    keep = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close']
            if c in df.columns]
    df = df[keep]
    df.index.name = 'date'
    return df


def eng_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成全部特征列，返回 df 副本"""
    X = df.copy()
    c = X['close']; h = X['high']; l = X['low']
    # 兼容 vol 和 volume 列名
    vol_col = 'volume' if 'volume' in X.columns else ('vol' if 'vol' in X.columns else None)
    v = X[vol_col] if vol_col else pd.Series(np.nan, index=X.index)

    # ── 收益率 ──
    X['r1'] = c.pct_change(1)
    X['r5'] = c.pct_change(5)
    X['r10'] = c.pct_change(10)
    X['r20'] = c.pct_change(20)
    X['lr1'] = np.log(c / c.shift(1))

    # ── 波动率 ──
    X['v10'] = X['r1'].rolling(10).std()
    X['v20'] = X['r1'].rolling(20).std()
    X['v60'] = X['r1'].rolling(60).std()
    X['vv'] = X['v10'] / X['v60'].clip(lower=1e-10)  # 波动率突变

    # ── ATR ──
    hl = h - l; hc = (h - c.shift(1)).abs(); lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(1)
    X['atr'] = tr.rolling(14).mean()
    X['atr_pct'] = X['atr'] / c.clip(lower=1e-10)

    # ── 均线乖离 ──
    for w in [5, 10, 20, 60, 120]:
        ma = c.rolling(w).mean()
        X[f'dm{w}'] = (c - ma) / ma.clip(lower=1e-10)

    # ── 布林带 ──
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    X['bbp'] = (c - mid) / (2 * sd + 1e-10)
    X['bbw'] = (4 * sd) / mid.clip(lower=1e-10)

    # ── RSI(14) ──
    delta = X['r1']
    g = delta.where(delta > 0, 0).rolling(14).mean()
    ls = (-delta.where(delta < 0, 0)).rolling(14).mean()
    X['rsi'] = 100 - 100 / (1 + g / (ls + 1e-10))

    # ── MACD ──
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    macd = e12 - e26
    X['macd'] = macd
    X['macds'] = macd.ewm(span=9).mean()
    X['macdh'] = macd - X['macds']

    # ── 成交量 ──
    X['vma5'] = v.rolling(5).mean()
    X['vma20'] = v.rolling(20).mean()
    X['vr'] = v / X['vma20'].clip(lower=1e-10)
    X['v5r'] = X['vma5'] / X['vma20'].clip(lower=1e-10)

    # ── 价格通道 ──
    X['hh20'] = h.rolling(20).max(); X['ll20'] = l.rolling(20).min()
    X['brk_h'] = (c >= X['hh20'].shift(1)).astype(float)
    X['brk_l'] = (c <= X['ll20'].shift(1)).astype(float)

    # ── 涨跌停检测 ──
    if 'pre_close' in X.columns:
        pct = (c - X['pre_close']) / X['pre_close']
    else:
        pct = X['r1']
    X['limup'] = (pct >= 0.095).astype(float)
    X['limdn'] = (pct <= -0.095).astype(float)

    # ── 振幅 ──
    X['rng'] = (h - l) / c.clip(lower=1e-10)

    # ── 收盘位置（日内） ──
    X['closepos'] = (c - l) / (h - l + 1e-10)

    return X
