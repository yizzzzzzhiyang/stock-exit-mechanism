#!/usr/bin/env python3
"""
股票退出机制 → 供应链选股
Streamlit 前端 — 一键启动，浏览器操作

启动: cd ~/供应链选股 && streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ── 页面配置 ──
st.set_page_config(
    page_title="供应链选股 v1.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 工具函数
# ============================================================

@st.cache_data(ttl=300)
def load_serenity_picks():
    """加载 Serenity 最新选股 JSON"""
    path = os.path.expanduser("~/public/serenity_picks.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    # 适配新旧两种 JSON 格式
    # 新格式（2026-07-09+）：top_sectors, top_picks, generated_at
    # 旧格式：sectors, rankings, date
    if 'top_sectors' in data or 'top_picks' in data:
        adapted = _adapt_v2_schema(data)
        return adapted
    return data


def _adapt_v2_schema(data: dict) -> dict:
    """将新格式 Serenity JSON 适配为面板能读的旧格式"""
    adapted = {
        'date': data.get('generated_at', '')[:10],
        'market': {},  # 新格式没有大盘概览
        'sectors': [],
        'rankings': [],
        'stocks': [],
    }

    # 板块
    for sec in data.get('top_sectors', []):
        adapted_sec = {
            'name': sec.get('sector_name', sec.get('sector_code', '?')),
            'gain': sec.get('combined_score', ''),
            'ai_analysis': sec.get('rationale', ''),
            'stocks': [],
        }
        # 新格式的个股在顶层 top_picks 里，按板块名匹配
        sec_name = adapted_sec['name']
        for pick in data.get('top_picks', []):
            if pick.get('sector_name', '') == sec_name:
                adapted_sec['stocks'].append({
                    'name': pick.get('name', pick.get('code', '?')),
                    'code': pick.get('code', ''),
                    'score': pick.get('funnel_score', 50),
                    'action': '买入' if pick.get('funnel_score', 0) >= 5 else '观察',
                    'stop_pct': f"-{pick.get('suggested_position_pct', 5)}%",
                })
        adapted['sectors'].append(adapted_sec)

    # 个股排名
    for i, pick in enumerate(data.get('top_picks', [])):
        adapted['rankings'].append({
            'rank': i + 1,
            'name': pick.get('name', '?'),
            'code': pick.get('code', ''),
            'score': pick.get('funnel_score', 50),
            'action': '买入' if pick.get('funnel_score', 0) >= 5 else '观察',
            'stop_pct': f"-{pick.get('suggested_position_pct', 5)}%",
            'position': f"{pick.get('suggested_position_pct', 5)}%",
        })

    adapted['stocks'] = adapted['rankings']
    return adapted


@st.cache_data(ttl=120)
def fetch_stock_data(stock, days=120):
    """获取日线数据：优先 NAS，不足时用 tushare 补齐"""
    import glob

    # 补全交易所后缀
    if '.' not in stock:
        code = int(stock)
        if 920000 <= code <= 929999:
            stock = f"{stock}.BJ"
        elif 600000 <= code <= 609999 or 688000 <= code <= 689999:
            stock = f"{stock}.SH"
        else:
            stock = f"{stock}.SZ"

    df = None

    # 1) 读 NAS
    try:
        nas_dir = '/Volumes/quant/stocks'
        if os.path.exists(nas_dir):
            files = sorted(glob.glob(os.path.join(nas_dir, 'daily_*.csv')))
            frames = []
            for f in files[-days:]:
                d = pd.read_csv(f, dtype={'ts_code': str})
                d = d[d['ts_code'] == stock]
                if len(d) > 0:
                    frames.append(d)
            if frames:
                df = pd.concat(frames, ignore_index=True)
    except:
        pass

    # 2) NAS 不足 → tushare 补齐
    if df is None or len(df) < 60:
        try:
            os.environ['no_proxy'] = 'tushare.pro,api.tushare.pro,*'
            import tushare as ts
            pro = ts.pro_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days+60)).strftime("%Y%m%d")
            ts_df = pro.daily(ts_code=stock, start_date=start, end_date=end,
                              fields='trade_date,open,high,low,close,vol')
            if ts_df is not None and len(ts_df) > 0:
                ts_df['trade_date'] = pd.to_datetime(ts_df['trade_date'])
                ts_df = ts_df.sort_values('trade_date')
                ts_df = ts_df.set_index('trade_date')
                ts_df.rename(columns={'vol': 'volume'}, inplace=True)
                return ts_df[['open','high','low','close','volume']]
        except:
            pass

    if df is None or len(df) < 20:
        # 3) 本地缓存 (sector-radar K线数据)
        try:
            cache_path = "/tmp/sector_radar_klines.json"
            if os.path.exists(cache_path):
                # stock 格式: "000001.SZ" → 纯码 "000001"
                pure_code = stock.split(".")[0]
                with open(cache_path) as f:
                    cache = json.load(f)
                if pure_code in cache:
                    rows = cache[pure_code]
                    local_df = pd.DataFrame(rows)
                    local_df["trade_date"] = pd.to_datetime(local_df["date"])
                    local_df = local_df.set_index("trade_date").sort_index()
                    local_df = local_df[["open", "high", "low", "close", "volume"]]
                    if len(local_df) >= 20:
                        return local_df
        except:
            pass

    if df is None or len(df) < 20:
        return None

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.drop_duplicates('trade_date').sort_values('trade_date')
    df = df.set_index('trade_date')
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.rename(columns={'vol': 'volume'}, inplace=True)
    return df


def calc_features(df):
    """从日线数据计算特征"""
    if df is None or len(df) < 60:
        return None

    c = df['close']
    ret = c.pct_change().dropna()

    # σ (20日波动率)
    sigma = ret.iloc[-20:].std() if len(ret) >= 20 else 0.02
    if sigma <= 0.001:
        sigma = 0.02

    # MA60
    ma60 = c.iloc[-60:].mean()

    # CUSUM 信号
    cusum_triggered = False
    if len(ret) >= 21:
        threshold = 1.5 * sigma
        pos_sum = 0
        for r in ret.iloc[-21:-1].values:
            pos_sum = max(0, pos_sum + r - sigma * 0.5)
            if pos_sum > threshold:
                pos_sum = 0
        pos_sum = max(0, pos_sum + ret.iloc[-1] - sigma * 0.5)
        cusum_triggered = pos_sum > threshold

    # 当前价格
    current = float(c.iloc[-1])

    # ATR
    h, l = df['high'], df['low']
    hl = h - l
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_val = float(tr.iloc[-14:].mean()) if len(tr) >= 14 else sigma * current

    return {
        'current': current,
        'sigma': sigma,
        'ma60': ma60,
        'cusum': cusum_triggered,
        'atr': atr_val,
        'date': df.index[-1],
        'n_days': len(df),
    }


def calc_exit_levels(entry_price, highest, sigma,
                     lower_mult=1.0, trail_mult=3.0):
    """计算三条退出线"""
    lower = entry_price * (1 - lower_mult * sigma)
    trail = highest * (1 - trail_mult * sigma)
    return {
        'lower': round(lower, 2),
        'trail': round(trail, 2),
        'lower_pct': round((lower / entry_price - 1) * 100, 1),
        'trail_pct': round((trail / highest - 1) * 100, 1),
    }


def load_portfolio():
    """加载持仓数据"""
    path = os.path.join(PROJECT_DIR, 'data', 'portfolio.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_portfolio(positions):
    """保存持仓数据"""
    path = os.path.join(PROJECT_DIR, 'data', 'portfolio.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


# ============================================================
# 页面主体
# ============================================================

st.title("📈 股票退出机制 v1.1")
st.caption("Serenity 1.0 选股 → CUSUM 择时 → 追损 3.0σ 退出")

# ── 侧边栏 ──
with st.sidebar:
    st.header("⚙️ 控制面板")

    if st.button("🔄 刷新全部数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # 参数显示
    st.subheader("📐 策略参数")
    st.metric("追损倍数", "3.0σ")
    st.metric("下屏障", "1.0σ")
    st.metric("最大持仓", "20天")
    st.metric("MA趋势过滤", "MA60")

    st.divider()

    # Serenity 状态
    picks = load_serenity_picks()
    if picks:
        st.success(f"✅ Serenity: {picks.get('date', '?')}")
    else:
        st.warning("⚠️ serenity_picks.json 不存在")

    st.divider()
    st.caption("供应链选股 v1.1 · 本地运行 · 手动操作")

# ── 主内容区：三个Tab ──
tab1, tab2, tab3 = st.tabs(["🎯 今日选股", "📊 买卖信号", "📋 我的持仓"])

# ============================================================
# Tab 1: 今日选股
# ============================================================
with tab1:
    if picks is None:
        st.info("还没运行 Serenity。请先执行: `cd ~/.hermes/projects/serenity-report/scripts && python3 pipeline.py`")
    else:
        # ── 市场概览 ──
        mkt = picks.get('market', {})
        if mkt:
            st.subheader(f"📊 市场概览 — {picks['date']}")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("上涨", f"{mkt.get('ups',0):,}只")
            k2.metric("下跌", f"{mkt.get('downs',0):,}只")
            k3.metric("涨停", f"{mkt.get('limit_ups',0)}只")
            k4.metric("成交额", f"{mkt.get('total_amount',0)}亿")
            k5.metric("涨跌比", f"{mkt.get('up_down_ratio',0):.2f}")
            k6.metric("中位数", f"{mkt.get('median_chg',0):+.2f}%")

        # ── 板块分析 ──
        sectors = picks.get('sectors', [])
        if sectors:
            st.divider()
            st.subheader("🔥 板块分析 & 供应链瓶颈")

            for sec in sectors:
                with st.expander(f"▸ {sec['name']}  {'+' + str(sec['gain']) + '%' if sec.get('gain') else ''}", expanded=(len(sectors)==1)):
                    col_left, col_right = st.columns([2, 1])

                    with col_left:
                        # AI 分析
                        ai = sec.get('ai_analysis', '')
                        if ai:
                            st.markdown(f"💬 **AI 解读**: {ai}")

                        # 供应链
                        chain = sec.get('supply_chain', [])
                        if chain:
                            st.markdown(f"🔗 **供应链**: {' → '.join(chain)}")

                    with col_right:
                        cycle = sec.get('cycle', '')
                        choke = sec.get('choke_point', '')
                        if cycle:
                            st.info(f"🔄 {cycle}")
                        if choke:
                            st.error(f"🔴 瓶颈: {choke}")

                    # 候选个股
                    sec_stocks = sec.get('stocks', [])
                    if sec_stocks:
                        st.markdown("##### 🎯 受益个股")
                        scols = st.columns(min(len(sec_stocks), 3))
                        for j, ss in enumerate(sec_stocks):
                            with scols[j % 3]:
                                sc = ss['score']
                                score_bg = "#dbeafe" if sc>=90 else "#fef3c7"
                                st.markdown(
                                    f"""<div style="background:{score_bg}; border-radius:8px; padding:12px; margin:4px 0;">
                                    <div style="font-weight:bold; font-size:15px; color:#1e40af;">{ss['name']}</div>
                                    <div style="font-size:12px; color:#6b7280;">{ss['code']} · GARP {sc:.0f}/100</div>
                                    <div style="font-size:13px; color:#374151; margin-top:4px;">🏭 {ss.get('business','')[:50]}</div>
                                    <div style="font-size:12px; color:#dc2626; margin-top:2px;">🔗 {ss.get('bottleneck','')[:50]}</div>
                                    <div style="font-size:12px; color:#059669; margin-top:2px;">🔥 {ss.get('trend_role','')[:50]}</div>
                                    <div style="margin-top:6px; font-size:12px;">
                                        <span style="background:#e5e7eb; padding:1px 6px; border-radius:4px;">{ss.get('action','')}</span>
                                        <span style="background:#fee2e2; padding:1px 6px; border-radius:4px; margin-left:4px;">止损{ss.get('stop_pct','')}</span>
                                    </div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    # 多空信号
                    bull = sec.get('bull_signals', [])
                    bear = sec.get('bear_signals', [])
                    if bull or bear:
                        st.divider()
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            for b in bull:
                                st.caption(f"✅ {b}")
                        with bc2:
                            for b in bear:
                                st.caption(f"❌ {b}")

        # ── 个股排名 ──
        rankings = picks.get('rankings', [])
        if rankings:
            st.divider()
            st.subheader("🏆 个股优选排名")
            st.dataframe(
                pd.DataFrame(rankings).set_index('rank'),
                column_config={
                    "name": "股票", "code": "代码", "score": st.column_config.NumberColumn("GARP", format="%.0f"),
                    "pe": "PE", "action": "建议", "stop_pct": "止损", "position": "仓位",
                },
                use_container_width=True,
                hide_index=False,
            )


# ============================================================
# Tab 2: 买卖信号
# ============================================================
with tab2:
    st.subheader("📊 退出机制信号")

    if picks is None:
        st.info("请先运行 Serenity")
    else:
        # 从新格式提取股票列表（兼容 v1 和 v2）
        stocks = picks.get('rankings', picks.get('stocks', []))
        if not stocks:
            st.warning("无选股数据")
        else:
            # 统一到 {code, name} 格式
            stock_list = []
            for s in stocks:
                stock_list.append({"code": s["code"], "name": s["name"]})
            stocks = stock_list
            rows = []
            for s in stocks:
                df = fetch_stock_data(s['code'])
                if df is None:
                    rows.append({
                        "股票": s['name'], "代码": s['code'],
                        "现价": "—", "波动率": "—", "下屏障": "—",
                        "追损线": "—", "MA60": "—",
                        "信号": "❓ 无数据", "原因": "行情获取失败"
                    })
                    continue

                feat = calc_features(df)
                if feat is None:
                    rows.append({
                        "股票": s['name'], "代码": s['code'],
                        "现价": "—", "波动率": "—", "下屏障": "—",
                        "追损线": "—", "MA60": "—",
                        "信号": "❓ 数据不足", "原因": "历史数据不够"
                    })
                    continue

                # 判断信号
                price_ok = feat['current'] > feat['ma60']
                cusum_ok = feat['cusum']

                if cusum_ok and price_ok:
                    signal = "🟢 **买入**"
                    reason = f"CUSUM触发 + 现价{feat['current']:.2f}>MA60({feat['ma60']:.2f})"
                elif cusum_ok and not price_ok:
                    signal = "🟡 待确认"
                    reason = f"CUSUM触发但现价{feat['current']:.2f}<MA60({feat['ma60']:.2f})"
                elif not cusum_ok and price_ok:
                    signal = "🟡 趋势向上"
                    reason = f"MA60上方，等待CUSUM触发"
                else:
                    signal = "⏸ 观望"
                    reason = f"现价<MA60，CUSUM未触发"

                exits = calc_exit_levels(feat['current'], feat['current'], feat['sigma'])

                rows.append({
                    "股票": s['name'], "代码": s['code'],
                    "现价": f"{feat['current']:.2f}",
                    "波动率": f"{feat['sigma']*100:.1f}%",
                    "下屏障": f"{exits['lower']} ({exits['lower_pct']}%)",
                    "追损线": f"{exits['trail']} ({exits['trail_pct']}%)",
                    "MA60": f"{feat['ma60']:.2f}",
                    "信号": signal,
                    "原因": reason,
                })

            if rows:
                df_sig = pd.DataFrame(rows)
                st.markdown("### 信号一览")
                for _, r in df_sig.iterrows():
                    if "买入" in r['信号']:
                        border, bg, tag_color = "#22c55e", "#f0fdf4", "#166534"
                    elif "待确认" in r['信号'] or "趋势向上" in r['信号']:
                        border, bg, tag_color = "#eab308", "#fefce8", "#854d0e"
                    else:
                        border, bg, tag_color = "#9ca3af", "#f9fafb", "#6b7280"

                    st.markdown(
                        f"""<div style="border:1px solid #e5e7eb; border-left:4px solid {border}; border-radius:8px; padding:14px; margin:8px 0; background:{bg};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:bold; font-size:16px; color:#111827;">{r['股票']}</span>
                            <span style="color:#6b7280; font-size:13px;">{r['代码']}</span>
                        </div>
                        <div style="margin-top:8px; font-size:14px; color:#374151;">
                            现价 <b style="color:#111827;">{r['现价']}</b>
                            &nbsp;｜&nbsp; 波动率 {r['波动率']}
                            &nbsp;｜&nbsp; MA60 {r['MA60']}
                        </div>
                        <div style="margin-top:4px; font-size:13px; color:#6b7280;">
                            下屏障 {r['下屏障']} &nbsp;｜&nbsp; 追损线 {r['追损线']}
                        </div>
                        <div style="margin-top:6px; font-size:14px; color:{tag_color}; font-weight:600;">
                            {r['信号']}
                            <span style="color:#6b7280; font-weight:400; font-size:13px;">&nbsp;— {r['原因']}</span>
                        </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ============================================================
# Tab 3: 我的持仓
# ============================================================
with tab3:
    st.subheader("📋 持仓管理")

    # ── 添加持仓 ──
    with st.expander("➕ 添加新持仓", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            new_code = st.text_input("股票代码", placeholder="如 688578.SH")
        with col_b:
            new_price = st.number_input("入场价格", min_value=0.01, value=100.0, step=0.01)
        with col_c:
            new_vol = st.number_input("买入数量（股）", min_value=100, value=100, step=100)

        if st.button("确认添加", use_container_width=True):
            if new_code:
                positions = load_portfolio()
                positions.append({
                    "code": new_code,
                    "name": "",
                    "entry_price": new_price,
                    "volume": new_vol,
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    "highest": new_price,
                })
                save_portfolio(positions)
                st.success(f"已添加 {new_code}")
                st.rerun()

    # ── 当前持仓 ──
    positions = load_portfolio()

    if not positions:
        st.info("暂无持仓。点击上方「添加新持仓」录入已买入的股票。")
    else:
        # 获取每只持仓的实时信号
        enriched = []
        for pos in positions:
            code = pos['code']
            df = fetch_stock_data(code)
            feat = calc_features(df) if df is not None else None

            current = feat['current'] if feat else pos['entry_price']
            sigma = feat['sigma'] if feat else 0.02
            highest = max(pos.get('highest', pos['entry_price']), current)

            # 更新最高价
            pos['highest'] = highest

            exits = calc_exit_levels(pos['entry_price'], highest, sigma)

            pnl_pct = (current / pos['entry_price'] - 1) * 100
            pnl_amt = (current - pos['entry_price']) * pos['volume']

            # 信号判断
            if current <= exits['lower']:
                sig = "🔴 止损触发"
                sig_detail = f"现价{current:.2f} ≤ 下屏障{exits['lower']}"
            elif current <= exits['trail']:
                sig = "🟠 追损触发"
                sig_detail = f"现价{current:.2f} ≤ 追损线{exits['trail']}"
            else:
                sig = "🟢 持有"
                sig_detail = f"追损线 {exits['trail']}，距离 {((current/exits['trail']-1)*100):+.1f}%"

            enriched.append({
                "code": code,
                "name": pos.get('name', code),
                "entry_price": pos['entry_price'],
                "current": current,
                "volume": pos['volume'],
                "entry_date": pos['entry_date'],
                "pnl_pct": pnl_pct,
                "pnl_amt": pnl_amt,
                "sigma": sigma,
                "lower": exits['lower'],
                "trail": exits['trail'],
                "signal": sig,
                "signal_detail": sig_detail,
            })

        save_portfolio(positions)  # 保存更新后的最高价

        # 逐个持仓卡片
        for i, p in enumerate(enriched):
            pnl_color = "#059669" if p['pnl_pct'] >= 0 else "#dc2626"
            if "止损" in p['signal']:
                sig_color, sig_bg = "#dc2626", "#fef2f2"
            elif "追损" in p['signal']:
                sig_color, sig_bg = "#ea580c", "#fff7ed"
            else:
                sig_color, sig_bg = "#059669", "#f0fdf4"

            col_main, col_del = st.columns([20, 1])
            with col_main:
                st.markdown(
                    f"""<div style="border:1px solid #e5e7eb; border-left:5px solid {sig_color}; border-radius:10px; padding:14px; margin:6px 0; background:#fff;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:bold; font-size:17px; color:#111827;">{p['code']}</span>
                            <span style="color:#6b7280; margin-left:8px;">{p['volume']}股</span>
                            <span style="color:#6b7280; margin-left:8px;">入场 {p['entry_date']}</span>
                        </div>
                        <div>
                            <span style="font-weight:bold; color:{pnl_color}; font-size:18px;">{p['pnl_pct']:+.1f}%</span>
                            <span style="font-weight:bold; color:{pnl_color}; margin-left:8px;">{p['pnl_amt']:+.0f}元</span>
                        </div>
                    </div>
                    <div style="margin-top:8px; font-size:14px; color:#374151;">
                        入场价 {p['entry_price']:.2f} → 现价 <b style="color:#111827;">{p['current']:.2f}</b>
                        &nbsp;｜&nbsp; 下屏障 <span style="color:#dc2626; font-weight:600;">{p['lower']}</span>
                        &nbsp;｜&nbsp; 追损线 <span style="color:#ea580c; font-weight:600;">{p['trail']}</span>
                    </div>
                    <div style="margin-top:6px; padding:4px 10px; background:{sig_bg}; border-radius:6px; font-size:14px; font-weight:600; color:{sig_color}; display:inline-block;">
                        {p['signal']} — {p['signal_detail']}
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑", key=f"del_{i}", help=f"删除 {p['code']}"):
                    positions.pop(i)
                    save_portfolio(positions)
                    st.rerun()

# ── 底部 ──
st.divider()
st.caption("本面板仅供参考，不构成投资建议。买卖决策由您手动执行。")
