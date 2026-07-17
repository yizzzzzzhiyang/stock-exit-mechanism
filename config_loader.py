"""
YAML 配置加载器

用法:
    from config_loader import load_config
    cfg = load_config('configs/default.yaml')
    cfg['strategy']['cusum_vol_mult']  # → 1.5
    cfg['barriers']                     # → [[2,1], [2,2], ...]
"""
import copy
import os
from typing import Any, Dict
import yaml


# ── 默认配置 ──
_DEFAULT_CONFIG: Dict[str, Any] = {
    'strategy': {
        'cusum_vol_mult': 1.5,
        'num_days': 20,
        'meta_threshold': 0.3,
    },
    'barriers': [[2, 1], [2, 2], [3, 1], [1, 1]],
    'cost': {
        'commission': 0.00025,
        'stamp_tax': 0.001,
        'slippage': 0.001,
    },
}


def load_config(path: str = '') -> Dict[str, Any]:
    """
    加载 YAML 配置文件，缺失的字段用默认值填充。

    参数:
        path: YAML 文件路径。为空时只返回默认配置。

    返回:
        dict: 完整的配置字典。
    """
    config = copy.deepcopy(_DEFAULT_CONFIG)

    if not path:
        return config

    # 尝试多个路径
    candidates = [
        path,
        os.path.join(os.path.dirname(__file__), path),
    ]
    # 如果 path 是相对路径，也尝试项目根目录
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.path.dirname(__file__), 'configs', path))

    resolved = None
    for c in candidates:
        if os.path.exists(c):
            resolved = c
            break

    if not resolved:
        raise FileNotFoundError(
            f"配置文件未找到。尝试了:\n  " + "\n  ".join(candidates)
        )

    with open(resolved, encoding='utf-8') as f:
        overrides = yaml.safe_load(f)

    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    递归合并字典：override 的值覆盖 base 的同名键。
    如果 base 里没有某个键，直接添加。
    """
    for key, val in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(val, dict)
        ):
            _deep_merge(base[key], val)
        else:
            base[key] = val
