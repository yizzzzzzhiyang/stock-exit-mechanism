#coding:gbk
"""
=====================================================================
  股票退出机制 v1.1 — 追损3.0σ 退出策略
=====================================================================

  版本演进:
    v1.0 (TB)     Triple-Barrier: 上屏障+下屏障+时间，三柱固定
    v1.1 (追损)   下屏障+追损线+时间，去上屏障→追损3.0σ

  入场: CUSUM 事件 + MA60趋势过滤 + 非涨停
  退出: 下屏障(1.0σ) 盘中实时 + 追损(3.0σ) 收盘确认 + 20天强平
  风控: 单日亏损5%熔断 | 最多持仓5只 | 每只1万元

  选股: Serenity 1.0 (盘后) → serenity_picks.json → QMT(盘前自动读取)
        如 JSON 不可用 → 回退到默认5只股票池

  回测验证 (2015-2026, 5只股票):
    牧原股份:  TB[2,1] SR=0.009 → 追损3.0σ SR=0.093 (+933%)
    中国平安:  TB[2,1] SR=-0.124 → 追损3.0σ SR=-0.061 (少亏51%)
    五粮液:    TB[2,1] SR=0.325 → 追损3.0σ SR=0.436 (+34%)
    三生国健:  TB[2,1] SR=0.427 → 追损3.0σ SR=0.664 (+56%)
    东方电气:  TB[2,1] SR=0.041 → 追损2.5σ SR=0.231 (+463%)

  QMT 使用: 模型交易 → 新建策略 → 粘贴全部代码 → 修改 account → 运行
  操作手册: 项目根目录/操作手册.md

  项目: /Users/yizhiyang/股票退出机制1.0/
=====================================================================
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta


# ============================================================
# 配置区 —— 修改你的资金账号
# ============================================================
CONFIG = {
    'account':         '你的资金账号',
    # ↓ 默认股票池（Serenity JSON 不可用时回退到此）
    'fallback_pool':   ['002714.SZ', '601318.SH', '000858.SZ',
                        '688336.SH', '600875.SH'],
    'picks_json':      os.path.expanduser('~/public/serenity_picks.json'),
    'max_json_age_days': 3,  # JSON超过3天不用
    'cash_per_stock':   10000,
    'max_positions':    5,
    'cusum_vol_mult':   1.5,
    'sigma_window':     20,
    'lower_mult':       1.0,
    'trail_mult':       3.0,
    'max_hold_days':    20,
    'ma_trend_window':  60,
    'daily_loss_limit': -0.05,
}


def load_serenity_picks():
    """
    从 Serenity 1.0 输出的 JSON 读取今日选股。
    文件不可用/过期 → 回退到默认股票池。
    返回: [股票代码列表]
    """
    path = CONFIG['picks_json']
    if not os.path.exists(path):
        print('[选股] serenity_picks.json 不存在，使用默认股票池')
        return CONFIG['fallback_pool']

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        print('[选股] JSON 解析失败，使用默认股票池')
        return CONFIG['fallback_pool']

    # 检查日期是否过期
    try:
        file_date = datetime.strptime(data.get('date', '2000-01-01'), '%Y-%m-%d')
        age = (datetime.now() - file_date).days
        if age > CONFIG['max_json_age_days']:
            print('[选股] JSON 已过期(%d天)，使用默认股票池' % age)
            return CONFIG['fallback_pool']
    except:
        pass

    picks = [s['code'] for s in data.get('stocks', [])]
    if not picks:
        print('[选股] JSON 内容为空，使用默认股票池')
        return CONFIG['fallback_pool']

    print('[选股] 来自 Serenity %s — %d只: %s' %
          (data.get('date', '?'), len(picks), ', '.join(picks)))
    return picks


# ============================================================
# CUSUM 事件检测
# ============================================================

def calc_cusum(prices, threshold_mult=1.5):
    if len(prices) < 20:
        return False
    rets = prices.pct_change().dropna()
    if len(rets) < 10:
        return False

    sigma = rets.iloc[-20:].std()
    if sigma <= 0.001:
        sigma = 0.02

    threshold = threshold_mult * sigma

    # 扫描过去20天，排除已经触发过的
    pos_sum = 0.0
    for r in rets.iloc[-21:-1].values:
        pos_sum = max(0, pos_sum + r - sigma * 0.5)
        if pos_sum > threshold:
            pos_sum = 0

    latest_r = rets.iloc[-1]
    pos_sum = max(0, pos_sum + latest_r - sigma * 0.5)
    return pos_sum > threshold


# ============================================================
# 数据获取 & 计算
# ============================================================

def get_daily_data(ContextInfo, stock, days=120):
    try:
        data = ContextInfo.get_market_data(
            ['close'], [stock],
            period='1d', count=days,
            dividend_type='front'
        )
        if data is None or stock not in data:
            return None
        closes = data[stock]['close']
        if len(closes) < 20:
            return None
        return closes
    except:
        return None


def calc_sigma(prices):
    if len(prices) < 5:
        return 0.02
    rets = prices.pct_change().dropna()
    sigma = rets.iloc[-20:].std() if len(rets) >= 20 else rets.std()
    return max(sigma, 0.005)


def is_limit_up(ContextInfo, stock):
    try:
        tick = ContextInfo.get_full_tick([stock])
        if tick is None or stock not in tick:
            return False
        info = tick[stock]
        last_price = info.get('lastPrice', 0)
        pre_close = info.get('preClose', 0)
        if last_price <= 0 or pre_close <= 0:
            return False
        return (last_price / pre_close - 1) >= 0.095
    except:
        return False


# ============================================================
# QMT 标准入口
# ============================================================

def init(ContextInfo):
    global G
    G = type('G', (), {})()

    G.positions = {}
    G.today_buys = 0
    G.today_pnl = 0.0
    G.today_entry_value = 0.0
    G.last_trade_day = None
    G.cache_sigma = {}
    G.cache_ma60 = {}

    # 从 Serenity JSON 加载今日选股（不可用则回退默认池）
    G.stock_pool = load_serenity_picks()

    ContextInfo.run_time('check_entries', '1nDay',
                         '1970-01-01 09:25:00', 'SH')
    ContextInfo.run_time('check_exits', '3nSecond',
                         '1970-01-01 09:30:05', 'SH')
    ContextInfo.run_time('check_trailing_at_close', '1nDay',
                         '1970-01-01 14:55:00', 'SH')
    ContextInfo.run_time('reset_daily', '1nDay',
                         '1970-01-01 15:10:00', 'SH')

    print('[股票退出机制 v1.1] 初始化完成')
    print('  股票池(%d只): %s' % (len(G.stock_pool), G.stock_pool))
    print('  追损参数: lower=%.1fσ  trail=%.1fσ  max_hold=%dd' %
          (CONFIG['lower_mult'], CONFIG['trail_mult'], CONFIG['max_hold_days']))


def check_entries(ContextInfo):
    global G
    today = datetime.now().strftime('%Y%m%d')

    if G.last_trade_day != today:
        G.today_buys = 0
        G.today_pnl = 0.0
        G.today_entry_value = 0.0
        G.last_trade_day = today
        G.cache_sigma = {}
        G.cache_ma60 = {}

    if G.today_entry_value > 0:
        loss_ratio = G.today_pnl / max(G.today_entry_value, CONFIG['cash_per_stock'])
        if loss_ratio <= CONFIG['daily_loss_limit']:
            print('[风控] 单日亏损%.1f%%，熔断' % (loss_ratio * 100))
            return

    if len(G.positions) >= CONFIG['max_positions']:
        return

    for stock in G.stock_pool:
        if stock in G.positions:
            continue
        if G.today_buys >= CONFIG['max_positions'] - len(G.positions):
            break

        prices = get_daily_data(ContextInfo, stock, days=120)
        if prices is None:
            continue

        if stock not in G.cache_sigma:
            G.cache_sigma[stock] = calc_sigma(prices)
        sigma = G.cache_sigma[stock]

        if stock not in G.cache_ma60:
            G.cache_ma60[stock] = prices.iloc[-60:].mean()
        ma60 = G.cache_ma60[stock]

        current_price = prices.iloc[-1]

        cusum_ok = calc_cusum(prices, CONFIG['cusum_vol_mult'])
        trend_ok = current_price > ma60
        not_limit = not is_limit_up(ContextInfo, stock)

        if cusum_ok and trend_ok and not_limit:
            vol = int(CONFIG['cash_per_stock'] / (current_price * 100)) * 100
            if vol < 100:
                continue

            passorder(23, 1101, CONFIG['account'], stock, 11, 0, vol,
                      '', 1, 'CUSUM入场', ContextInfo)

            G.positions[stock] = {
                'entry_price': current_price,
                'highest': current_price,
                'entry_day': today,
                'volume': vol,
                'sigma': sigma,
            }
            G.today_buys += 1
            G.today_entry_value += CONFIG['cash_per_stock']

            print('[入场] %s @%.2f  σ=%.1f%%  MA60=%.2f  CUSUM触发' %
                  (stock, current_price, sigma * 100, ma60))


def check_exits(ContextInfo):
    global G
    for stock in list(G.positions.keys()):
        pos = G.positions[stock]
        try:
            tick = ContextInfo.get_full_tick([stock])
            if tick is None or stock not in tick:
                continue
            current = tick[stock].get('lastPrice', 0)
            if current <= 0:
                continue
        except:
            continue

        lower = pos['entry_price'] * (1 - CONFIG['lower_mult'] * pos['sigma'])
        if current <= lower:
            execute_sell(ContextInfo, stock, current, '下屏障止损')


def check_trailing_at_close(ContextInfo):
    global G
    for stock in list(G.positions.keys()):
        pos = G.positions[stock]
        try:
            tick = ContextInfo.get_full_tick([stock])
            if tick is None or stock not in tick:
                continue
            current = tick[stock].get('lastPrice', 0)
            if current <= 0:
                continue
        except:
            continue

        if current > pos['highest']:
            pos['highest'] = current

        sigma = pos['sigma']
        trail = pos['highest'] * (1 - CONFIG['trail_mult'] * sigma)

        try:
            entry_dt = datetime.strptime(pos['entry_day'], '%Y%m%d')
            days_held = (datetime.now() - entry_dt).days
        except:
            days_held = 0

        if current <= trail:
            execute_sell(ContextInfo, stock, current,
                         '追损%.1f%%回撤' % (CONFIG['trail_mult'] * sigma * 100))
        elif days_held >= CONFIG['max_hold_days']:
            execute_sell(ContextInfo, stock, current, '超时%d天平仓' % days_held)


def reset_daily(ContextInfo):
    global G
    G.today_buys = 0
    G.today_pnl = 0.0
    G.today_entry_value = 0.0
    print('[盘后] 持仓%d只' % len(G.positions))


def execute_sell(ContextInfo, stock, price, reason):
    global G
    pos = G.positions[stock]
    vol = pos['volume']

    passorder(23, 1102, CONFIG['account'], stock, 11, 0, vol,
              '', 1, reason, ContextInfo)

    pnl_pct = (price / pos['entry_price'] - 1) * 100
    pnl_amt = (price - pos['entry_price']) * vol
    G.today_pnl += pnl_amt

    print('[卖出] %s @%.2f  盈亏%+.1f%% %+.0f元  %s' %
          (stock, price, pnl_pct, pnl_amt, reason))

    del G.positions[stock]


def handlebar(ContextInfo):
    pass
