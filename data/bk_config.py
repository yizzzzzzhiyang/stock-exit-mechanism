"""
BK 分类加载与屏障配置优化模块

数据来源: /Users/yizhiyang/.serenity_data/sectors/concept/
每个 BKXXX.csv 包含该概念板块的股票代码列表
_sectors.csv 提供 BK 代码到名称的映射

功能:
  1. 加载 BK 分类数据
  2. 对每个概念板块推荐最优屏障配置
  3. 给定任意股票代码，返回推荐配置
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

from config import get_concept_dir, get_industry_dir, get_fundamental_file, setup_proxy_bypass

warnings.filterwarnings('ignore')
setup_proxy_bypass()

BK_DIR = get_concept_dir()

# ── 股票代码市场后缀映射 ──
# 沪市: 600/601/603/605/688 开头 → .SH
# 深市: 000/001/002/003/300/301 开头 → .SZ
# 北交所: 8xx 开头 → .BJ
def code_to_ts(code: str) -> str:
    """6位股票代码 → tushare 格式"""
    code = code.strip()
    if len(code) < 6:
        return code
    prefix = code[:3]
    if prefix in ('600','601','603','605','688','689'):
        return f'{code}.SH'
    elif prefix in ('000','001','002','003','300','301'):
        return f'{code}.SZ'
    elif prefix in ('830','831','832','833','834','835','836','837','838','839',
                    '870','871','872','873','874','875','876','877','878','879',
                    '920','921','922','923','924'):
        return f'{code}.BJ'
    return code


def load_bk_mapping(fundamental_path: str = None) -> dict:
    """
    加载 BK 分类, 返回:
      stock_to_bk: { '000001.SZ': ['银行', 'MSCI中国', ...], ... }
      bk_to_name:  { 'BK0477': '银行', ... }
      bk_to_stocks: { 'BK0477': ['000001.SZ', '600000.SH', ...], ... }
    
    如果 concept BK 查不到, 会尝试 fundamental 数据的行业分类
    """
    # ── 1. 加载 concept BK ──
    sectors_file = os.path.join(BK_DIR, '_sectors.csv')
    bk_names = {}
    if os.path.exists(sectors_file):
        df = pd.read_csv(sectors_file)
        for _, row in df.iterrows():
            bk_names[str(row['code']).strip()] = str(row['name']).strip()

    stock_to_bk = {}
    bk_to_stocks = {}

    for fname in os.listdir(BK_DIR):
        if not fname.startswith('BK') or not fname.endswith('.csv'):
            continue
        if fname == '_sectors.csv':
            continue
        bk_code = fname.replace('.csv', '')
        fpath = os.path.join(BK_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            if 'code' not in df.columns:
                continue
            codes = df['code'].dropna().astype(str).str.strip().tolist()
            ts_codes = [code_to_ts(c) for c in codes if len(c) >= 6]
            bk_to_stocks[bk_code] = ts_codes
            name = bk_names.get(bk_code, '')
            for tc in ts_codes:
                if tc not in stock_to_bk:
                    stock_to_bk[tc] = []
                stock_to_bk[tc].append((bk_code, name))
        except Exception:
            continue

    # ── 2. 对概念BK找不到的股票, 用行业BK补充 ──
    ind_dir = get_industry_dir()
    if os.path.exists(ind_dir):
        ind_names = {}
        ind_file = os.path.join(ind_dir, '_sectors.csv')
        if os.path.exists(ind_file):
            df = pd.read_csv(ind_file)
            for _, row in df.iterrows():
                ind_names[str(row['code']).strip()] = str(row['name']).strip()
        for fname in os.listdir(ind_dir):
            if not fname.startswith('BK') or not fname.endswith('.csv'):
                continue
            if fname == '_sectors.csv':
                continue
            bk_code = fname.replace('.csv', '')
            fpath = os.path.join(ind_dir, fname)
            try:
                df = pd.read_csv(fpath)
                if 'code' not in df.columns:
                    continue
                codes = df['code'].dropna().astype(str).str.strip().tolist()
                ts_codes = [code_to_ts(c) for c in codes if len(c) >= 6]
                name = ind_names.get(bk_code, '')
                for tc in ts_codes:
                    if tc not in stock_to_bk:
                        stock_to_bk[tc] = []
                    # 只添加概念BK中没有的股票
                    existing_bks = [b for b, _ in stock_to_bk[tc]]
                    if bk_code not in existing_bks:
                        stock_to_bk[tc].append((bk_code, name))
            except Exception:
                continue

    return stock_to_bk, bk_names, bk_to_stocks


def _get_industry_from_fundamental(ts_code: str) -> str:
    """从基本面数据获取行业分类"""
    try:
        fpath = get_fundamental_file()
        if not os.path.exists(fpath):
            return ''
        df = pd.read_csv(fpath)
        # 股票代码在CSV中的格式可能是 "000001" 或 "000001.SZ"
        raw_code = ts_code.split('.')[0]
        # CSV 中代码是整数格式
        int_code = int(raw_code)
        match = df[df['股票代码'] == int_code]
        if len(match) > 0:
            industry = str(match.iloc[0].get('所处行业', ''))
            return industry
        return ''
    except Exception:
        return ''


# ── 屏障配置库 ──
# 基于回测经验, 按波动率+风格分类
BARRIER_PRESETS = {
    # 低波动/大盘/蓝筹
    '低波动价值': {'pt_sl': [2, 2], 'num_days': 30, 'desc': '对称屏障, 给够时间'},
    '银行':       {'pt_sl': [2, 2], 'num_days': 35, 'desc': '银行波动小, 需要耐心'},
    '保险':       {'pt_sl': [2, 2], 'num_days': 30, 'desc': '同银行'},
    '白酒':       {'pt_sl': [2, 1.5], 'num_days': 30, 'desc': '长趋势, 让利润跑'},
    
    # 中波动/周期
    '强周期':     {'pt_sl': [1, 1], 'num_days': 15, 'desc': '紧屏障快跑, 牧原验证'},
    '农业':       {'pt_sl': [1, 1], 'num_days': 15, 'desc': '猪周期同'},
    '化工':       {'pt_sl': [1.5, 1], 'num_days': 20, 'desc': '化工周期, 适度收紧'},
    '有色':       {'pt_sl': [1.5, 1], 'num_days': 20, 'desc': '同化工'},
    
    # 高波动/成长
    '科技成长':   {'pt_sl': [2.5, 1], 'num_days': 25, 'desc': '高波动+趋势, 宽止盈紧止损'},
    '半导体':     {'pt_sl': [3, 1], 'num_days': 20, 'desc': '高波动趋势型'},
    '芯片':       {'pt_sl': [3, 1], 'num_days': 20, 'desc': '同半导体'},
    '新能源':     {'pt_sl': [2.5, 1.5], 'num_days': 25, 'desc': '中等偏上波动'},
    '医药':       {'pt_sl': [2, 1.5], 'num_days': 25, 'desc': '医药成长+波动'},
    
    # 均值回复型
    '消费':       {'pt_sl': [1.5, 1.5], 'num_days': 25, 'desc': '均值回复, 对称屏障'},
    '家电':       {'pt_sl': [1.5, 1.5], 'num_days': 25, 'desc': '同消费'},
    '食品':       {'pt_sl': [1.5, 1.5], 'num_days': 25, 'desc': '同消费'},
    
    # 防御型
    '公用事业':   {'pt_sl': [2, 2], 'num_days': 45, 'desc': '低波动长持仓'},
    '水电':       {'pt_sl': [2, 2], 'num_days': 45, 'desc': '同公用事业'},
    '交通运输':   {'pt_sl': [2, 2], 'num_days': 35, 'desc': '稳定型'},
    
    # 通用默认
    '默认':       {'pt_sl': [2, 2], 'num_days': 20, 'desc': '通用均衡配置'},
}

# BK 名称到配置键的映射
BK_CONFIG_MAP = {
    # === 银行保险 ===
    '银行': '银行', '股份制银行': '银行', '城商行': '银行',
    
    # === 周期农业 ===
    '猪肉': '强周期', '养殖': '强周期', '饲料': '强周期', '农业': '农业',
    '农牧饲渔': '农业', '鸡肉概念': '强周期',
    
    # === 化工有色 ===
    '化工': '化工', '化肥': '化工', '氟化工': '化工', '磷化工': '化工',
    '有色': '有色', '黄金': '有色', '稀土': '有色', '钢铁': '有色',
    
    # === 科技 ===
    '半导体': '半导体', '芯片': '芯片', '光刻机': '半导体',
    '集成电路': '半导体', '存储芯片': '半导体',
    'AI': '科技成长', '人工智能': '科技成长', '算力': '科技成长',
    '机器人': '科技成长', '数字经济': '科技成长',
    '新能源': '新能源', '新能源车': '新能源', '锂电池': '新能源',
    '光伏': '新能源', '风电': '新能源',
    
    # === 医药 ===
    '医药': '医药', '创新药': '医药', '医疗器械': '医药',
    '生物医药': '医药', '中药': '医药', 'CRO': '医药',
    
    # === 消费 ===
    '白酒': '白酒', '食品': '食品', '饮料': '消费',
    '家电': '家电', '消费电子': '消费',
    '调味品': '食品', '乳业': '食品',
    
    # === 防御 ===
    '水电': '水电', '电力': '公用事业', '燃气': '公用事业',
    '公路': '交通运输', '铁路': '交通运输', '机场': '交通运输',
    
    # === 大盘蓝筹 ===
    '超级品牌': '低波动价值', '行业龙头': '低波动价值',
}


def get_recommended_config(ts_code: str) -> dict:
    """根据股票代码查询推荐的屏障配置"""
    stock_to_bk, bk_names, _ = load_bk_mapping()
    
    bk_info = stock_to_bk.get(ts_code, [])
    bk_names_list = [name for _, name in bk_info if name]
    
    if not bk_names_list:
        industry = _get_industry_from_fundamental(ts_code)
        if industry:
            bk_names_list = [industry]
    
    if not bk_names_list:
        return {**BARRIER_PRESETS['默认'], 'reason': '未找到BK分类, 使用默认'}
    
    combined = ' '.join(bk_names_list)
    
    # ── 精确行业匹配（高优先级） ──
    # 银行类
    if any(w in combined for w in ['银行','股份制银行']):
        return {**BARRIER_PRESETS['银行'], 'reason': f'银行类 → 银行配置'}
    # 保险/证券/金融
    if any(w in combined for w in ['保险','证券','信托','期货']):
        return {**BARRIER_PRESETS['银行'], 'reason': f'金融类 → 银行配置'}
    # 医药
    if any(w in combined for w in ['创新药','CRO','CAR-T','医药','生物医药','化学制药','中药','医疗器械']):
        return {**BARRIER_PRESETS['医药'], 'reason': f'医药类 → 医药配置'}
    # 白酒
    if '白酒' in combined:
        return {**BARRIER_PRESETS['白酒'], 'reason': f'白酒 → 白酒配置'}
    
    # ── K 线波动率特征分类（中优先级） ──
    # 新能源（优先于"周期股"）
    if any(w in combined for w in ['新能源','锂','光伏','风电','电池','麒麟电池','宁组合']):
        return {**BARRIER_PRESETS['新能源'], 'reason': f'新能源类 → 新能源配置'}
    # 半导体/芯片
    if any(w in combined for w in ['半导体','芯片','集成电路','光刻','存储芯片']):
        return {**BARRIER_PRESETS['科技成长'], 'reason': f'科技类 → 科技成长配置'}
    # 农业/养殖/猪
    if any(w in combined for w in ['养殖业','猪肉','农牧饲渔','饲料','农业']):
        return {**BARRIER_PRESETS['强周期'], 'reason': f'农业/养殖 → 强周期配置'}
    # 化工/有色/钢铁
    if any(w in combined for w in ['化工','有色','钢铁','化学原料','化学制品','化学纤维','农药']):
        return {**BARRIER_PRESETS['化工'], 'reason': f'化工/有色 → 化工配置'}
    # 家电/消费
    if any(w in combined for w in ['白色家电','家用电器','食品','饮料','乳业','调味']):
        return {**BARRIER_PRESETS['消费'], 'reason': f'消费类 → 消费配置'}
    # 公用事业
    if any(w in combined for w in ['电力','水电','燃气','自来水','环保','交通运输','公路','铁路','机场','港口']):
        return {**BARRIER_PRESETS['公用事业'], 'reason': f'公用事业 → 防御配置'}
    
    # ── 通用关键词匹配 ──
    for keyword, config_key in BK_CONFIG_MAP.items():
        if keyword in combined:
            config = BARRIER_PRESETS.get(config_key, BARRIER_PRESETS['默认'])
            return {**config, 'reason': f'{combined[:60]}... → {config_key}'}
    
    # ── 二次匹配 ──
    if any(w in combined for w in ['医','药','生物','疗']):
        return {**BARRIER_PRESETS['医药'], 'reason': f'医疗健康 → 医药配置'}
    if any(w in combined for w in ['科技','电子','算力','AI','机器人','软件']):
        return {**BARRIER_PRESETS['科技成长'], 'reason': f'科技类 → 科技成长配置'}
    if any(w in combined for w in ['消费','家电','白酒','食品']):
        return {**BARRIER_PRESETS['消费'], 'reason': f'消费类 → 消费配置'}
    # 仅当有多个周期相关概念时才归类为周期
    cycle_kw = ['周期','矿','煤','钢铁','有色']
    # '金'单独处理：排除"基金"中的"金"
    if '金' in combined and '基金' not in combined:
        cycle_kw.append('金')
    cycle_count = sum(1 for w in cycle_kw if w in combined)
    if cycle_count >= 2:
        return {**BARRIER_PRESETS['强周期'], 'reason': f'周期类(多个信号) → 强周期配置'}
    
    return {**BARRIER_PRESETS['默认'], 'reason': f'{combined[:50]} → 默认配置'}


def show_stock_bk(ts_code: str):
    """展示股票的 BK 分类和推荐配置"""
    stock_to_bk, bk_names, _ = load_bk_mapping()
    bks = stock_to_bk.get(ts_code, [])
    
    print(f'\n===== {ts_code} ====')
    if not bks:
        print('  BK分类: 未找到')
    else:
        print(f'  所属 {len(bks)} 个概念板块:')
        for bk in bks:
            name = bk_names.get(bk, '(无名称)')
            print(f'    {bk} = {name}')
    
    config = get_recommended_config(ts_code)
    print(f'\n  推荐屏障配置:')
    print(f"    pt_sl={config['pt_sl']}  num_days={config['num_days']}")
    print(f"    原因: {config['reason']}")
    print(f"    说明: {config.get('desc', '')}")
    return config


if __name__ == '__main__':
    # 测试几个股票
    for code in ['002714.SZ', '000001.SZ', '600519.SH', '300750.SZ', '688578.SH',
                 '600036.SH', '000333.SZ', '601318.SH', '600900.SH', '002415.SZ']:
        show_stock_bk(code)
