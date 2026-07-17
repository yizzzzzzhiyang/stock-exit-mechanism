"""
统一配置模块

所有路径和环境相关配置集中管理。
通过环境变量覆盖默认值，消除硬编码路径。

环境变量:
  SERENITY_DATA_DIR    — serenity 数据根目录 (默认 ~/.serenity_data)
  TUSHARE_TOKEN        — tushare token (默认从 ~/.tushare/token 读取)
  EXIT_MECHANISM_DIR   — 项目根目录 (默认自动检测)
"""
import os
from typing import List

# ── 项目根目录 ──
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_project_dir() -> str:
    """项目根目录"""
    return os.environ.get('EXIT_MECHANISM_DIR', _PROJECT_DIR)


# ── Serenity 数据目录 ──
def get_serenity_dir() -> str:
    """serenity 数据根目录"""
    return os.environ.get(
        'SERENITY_DATA_DIR',
        os.path.expanduser('~/.serenity_data')
    )


def get_sectors_dir() -> str:
    """BK 分类数据目录"""
    return os.path.join(get_serenity_dir(), 'sectors')


def get_concept_dir() -> str:
    """概念板块数据目录"""
    return os.path.join(get_sectors_dir(), 'concept')


def get_industry_dir() -> str:
    """行业板块数据目录"""
    return os.path.join(get_sectors_dir(), 'industry')


def get_fundamental_dir() -> str:
    """基本面数据目录"""
    return os.path.join(get_serenity_dir(), 'fundamental')


def get_fundamental_file(filename: str = 'stock_yjbb_em_20251231.csv') -> str:
    """基本面数据文件路径"""
    return os.path.join(get_fundamental_dir(), filename)


# ── Tushare ──
def get_tushare_token() -> str:
    """获取 tushare token"""
    token = os.environ.get('TUSHARE_TOKEN', '')
    if token:
        return token
    # 回退到 ~/.tushare/token
    token_file = os.path.expanduser('~/.tushare/token')
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()
    return ''


# ── 代理设置 ──
def setup_proxy_bypass() -> None:
    """设置 tushare 直连，绕过系统代理"""
    os.environ['no_proxy'] = 'tushare.pro,api.tushare.pro,*'


# ── 验证 ──
def validate() -> List[str]:
    """验证关键路径和配置，返回问题列表"""
    issues = []
    checks = {
        '项目目录': get_project_dir(),
        'Serenity 数据目录': get_serenity_dir(),
        '概念板块目录': get_concept_dir(),
        '行业板块目录': get_industry_dir(),
    }
    for name, path in checks.items():
        if not os.path.exists(path):
            issues.append(f'⚠ {name} 不存在: {path}')

    token = get_tushare_token()
    if not token:
        issues.append('⚠ TUSHARE_TOKEN 未设置，也未找到 ~/.tushare/token')

    return issues
