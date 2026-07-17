"""
CUSUM 滤波器单元测试

覆盖场景：
  - 累计变化超过阈值 → 产生事件
  - 累计变化未超阈值 → 无事件
  - 基于波动率的 CUSUM
  - 空序列
"""
import numpy as np
import pandas as pd
import pytest

from labeling.cusum import cusum_filter, cusum_filter_vol


class TestCusumFilter:
    """基础 CUSUM 滤波器测试"""

    def test_detects_upward_movement(self):
        """持续上涨超过阈值 → 产生事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 从 100 涨到 107，累计 +7%
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0,
                           105.0, 106.0, 107.0, 108.0, 109.0], index=dates)

        events = cusum_filter(close, threshold=0.05)  # 5% 阈值

        assert len(events) > 0  # 应该有事件

    def test_no_movement_no_events(self):
        """价格不动 → 无事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        close = pd.Series([100.0] * 10, index=dates)

        events = cusum_filter(close, threshold=0.01)

        assert len(events) == 0  # 不动就没有事件

    def test_detects_downward_movement(self):
        """持续下跌超过阈值 → 产生事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 从 100 跌到 93，累计 -7%
        close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0,
                           95.0, 94.0, 93.0, 92.0, 91.0], index=dates)

        events = cusum_filter(close, threshold=0.05)

        assert len(events) > 0

    def test_small_threshold_noise(self):
        """小幅波动未超阈值 → 无事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 在 100 附近小幅震荡 ±1%
        close = pd.Series([100.0, 101.0, 99.0, 100.0, 101.0,
                           99.0, 100.0, 101.0, 99.0, 100.0], index=dates)

        events = cusum_filter(close, threshold=0.05)  # 5% 阈值

        assert len(events) == 0  # 波动太小

    def test_empty_series(self):
        """空序列 → 空结果"""
        close = pd.Series([], dtype=float)
        events = cusum_filter(close, threshold=0.02)
        assert len(events) == 0

    def test_single_element(self):
        """单元素序列 → 空结果（无法计算 diff）"""
        close = pd.Series([100.0], index=pd.date_range('2024-01-01', periods=1))
        events = cusum_filter(close, threshold=0.02)
        assert len(events) == 0


class TestCusumFilterVol:
    """基于波动率的 CUSUM 测试"""

    def test_detects_movement_exceeding_vol(self):
        """涨幅超过 vol_mult × vol_pct → 产生事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        close = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0,
                           110.0, 112.0, 114.0, 116.0, 118.0], index=dates)
        # 波动率 2%
        vol = pd.Series([0.02] * 10, index=dates)

        events = cusum_filter_vol(close, vol, vol_mult=1.5)
        # 阈值 = 1.5 × 2% = 3%
        # 第2天涨2% → 没到3%
        # 第3天累计涨4% → 超3% → 事件

        assert len(events) >= 1

    def test_flat_vol_mult_large_no_events(self):
        """大幅波动率倍数 → 阈值极大 → 无事件"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0,
                           101.0, 102.0, 101.0, 100.0, 101.0], index=dates)
        vol = pd.Series([0.02] * 10, index=dates)

        events = cusum_filter_vol(close, vol, vol_mult=10.0)
        # 阈值 = 10 × 2% = 20%，价格波动最多 2%

        assert len(events) == 0

    def test_empty_series(self):
        """空序列 → 空结果"""
        close = pd.Series([], dtype=float)
        vol = pd.Series([], dtype=float)
        events = cusum_filter_vol(close, vol)
        assert len(events) == 0
