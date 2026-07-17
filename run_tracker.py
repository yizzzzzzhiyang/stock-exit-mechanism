"""
实验跟踪器 —— 每次回测自动记录参数 + 结果到 CSV

用法:
    from run_tracker import save_run, load_runs

    # 回测结束后调用
    save_run(
        params={'cusum_vol_mult': 1.5, 'num_days': 20, ...},
        metrics={'sharpe': 0.45, 'max_dd': -0.12, ...},
        stock_code='000001.SZ',
    )

    # 查看所有历史记录
    df = load_runs()
    df.sort_values('sharpe', ascending=False)
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ── 存储目录 ──
_RUNS_FILE: str = ''  # 延迟初始化


def _get_runs_file() -> str:
    """获取 runs.csv 路径，延迟初始化以支持 main.py 调用"""
    global _RUNS_FILE
    if not _RUNS_FILE:
        _RUNS_FILE = str(Path(__file__).parent / 'runs.csv')
    return _RUNS_FILE


def _generate_run_id(params: Dict[str, Any]) -> str:
    """
    生成唯一 run_id：参数哈希前 8 位 + 时间戳

    用哈希保证相同参数产出相同 run_id，方便去重。
    """
    param_json = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.md5(param_json.encode()).hexdigest()[:8]
    ts = datetime.now().strftime('%m%d_%H%M')
    return f"{ts}_{h}"


# ── 固定列模式 ──
# 所有记录共享相同列，确保 CSV 格式一致
_COLUMNS: list = [
    'run_id', 'timestamp', 'stock', 'date_range', 'config',
    # 参数
    'cusum_vol_mult', 'num_days', 'meta_threshold', 'barriers',
    # 回测指标
    'sharpe', 'max_dd', 'win_rate', 'profit_factor',
    'dsr', 'num_trades', 'best_baseline_sr',
]


def save_run(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    stock_code: str = '',
    date_range: str = '',
    config_file: str = '',
) -> str:
    """
    保存一次回测记录。

    参数:
        params: 策略参数字典（cusum_vol_mult, num_days, 等）
        metrics: 回测指标（sharpe, max_dd, win_rate, 等）
        stock_code: 股票代码
        date_range: 日期范围字符串
        config_file: 使用的配置文件路径

    返回:
        str: run_id
    """
    run_id = _generate_run_id(params)

    # 用固定列构建记录，缺失字段填空字符串
    record: Dict[str, Any] = {col: '' for col in _COLUMNS}
    record.update({
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'stock': stock_code,
        'date_range': date_range,
        'config': config_file,
    })

    # 扁平化参数（只写入固定列中的字段）
    for k, v in params.items():
        if k not in _COLUMNS:
            continue
        if isinstance(v, (list,)):
            record[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, (int, float, str, bool)):
            record[k] = v
        else:
            record[k] = str(v)

    # 加入回测指标（只写入固定列中的字段）
    for k, v in metrics.items():
        if k in _COLUMNS:
            record[k] = v

    # 读现有 CSV 判断是否需要写表头
    runs_file = _get_runs_file()
    write_header = not os.path.exists(runs_file)

    df = pd.DataFrame([record])
    df.to_csv(runs_file, mode='a', header=write_header, index=False, encoding='utf-8')

    return run_id


def load_runs() -> pd.DataFrame:
    """
    加载所有历史回测记录。

    返回:
        DataFrame: 按时间戳降序排列
    """
    runs_file = _get_runs_file()
    if not os.path.exists(runs_file):
        return pd.DataFrame()

    df = pd.read_csv(runs_file, encoding='utf-8')

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    return df


def print_summary(top_n: int = 10, metric: str = 'sharpe') -> None:
    """
    打印历史回测摘要。

    参数:
        top_n: 显示前 N 条
        metric: 排序指标
    """
    df = load_runs()
    if df.empty:
        print("暂无历史回测记录。")
        return

    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)

    print(f"\n{'='*70}")
    print(f"历史回测记录（按 {metric} 降序，前 {top_n} 条）")
    print(f"{'='*70}")

    cols = [c for c in ['run_id', 'stock', 'config', 'sharpe', 'max_dd',
                         'win_rate', 'profit_factor', 'num_trades']
            if c in df.columns]
    display = df[cols].head(top_n)

    for _, row in display.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                parts.append(f"{c}={v:.4f}")
            else:
                parts.append(f"{c}={v}")
        print("  |  ".join(parts))

    print(f"\n共 {len(df)} 条记录\n")
