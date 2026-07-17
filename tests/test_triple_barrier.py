"""
Triple-Barrier 标签生成器单元测试

覆盖场景：
  - 价格触及止盈线 → 标签 +1
  - 价格触及止损线 → 标签 -1
  - 价格未触及任何屏障 → 标签 0（时间到期）
  - 空事件 → 空结果
  - 垂直屏障边界
"""
import numpy as np
import pandas as pd
import pytest

from labeling.triple_barrier import add_vertical_barrier, apply_triple_barrier


class TestAddVerticalBarrier:
    """垂直屏障（时间屏障）测试"""

    def test_basic_vertical_barrier(self):
        """给定事件和价格序列，返回正确的垂直屏障时间戳"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        close = pd.Series(np.arange(10, 20), index=dates)
        t_events = pd.DatetimeIndex([dates[0], dates[3]])

        t1 = add_vertical_barrier(t_events, close, num_days=5)

        assert len(t1) == 2               # 两个事件都有屏障
        assert t1.iloc[0] == dates[5]     # 事件0 + 5天 → 第5天
        assert t1.iloc[1] == dates[8]     # 事件3 + 5天 → 第8天（但只有10天，所以 min(8, 9)=8）

    def test_vertical_barrier_at_end(self):
        """事件靠近序列末尾，垂直屏障不应越界"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        close = pd.Series(np.arange(10, 15), index=dates)
        t_events = pd.DatetimeIndex([dates[3]])  # 倒数第2天入场

        t1 = add_vertical_barrier(t_events, close, num_days=20)

        assert len(t1) == 1
        # 20天超出范围 → 应该停在最后一天
        assert t1.iloc[0] == dates[-1]

    def test_empty_events(self):
        """无事件 → 空 Series"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        close = pd.Series(np.arange(10, 15), index=dates)
        t_events = pd.DatetimeIndex([])

        t1 = add_vertical_barrier(t_events, close, num_days=5)

        assert len(t1) == 0


class TestApplyTripleBarrier:
    """三柱法标注测试"""

    def test_hits_upper_barrier(self):
        """价格先突破上屏障 → 标签 +1"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 价格稳步上升：10 → 10.5 → 11.5 → ... 第4天涨到13（+30%）
        close = pd.Series([10.0, 10.5, 11.5, 13.0, 13.5,
                           14.0, 14.5, 15.0, 15.5, 16.0], index=dates)
        vol = pd.Series([0.10] * 10, index=dates)  # 10% 波动率
        t_events = pd.DatetimeIndex([dates[0]])

        # pt_sl=[2,1] → 止盈=20%, 止损=10%
        result = apply_triple_barrier(close, t_events, pt_sl=[2, 1],
                                       vol=vol, num_days=20, detect_limits=False)

        assert result.loc[dates[0], 'bin'] == 1    # 触止盈
        assert result.loc[dates[0], 'ret'] > 0     # 正收益

    def test_hits_lower_barrier(self):
        """价格先突破下屏障 → 标签 -1"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 价格持续下跌：10 → 9.5 → 8.5 → ...
        close = pd.Series([10.0, 9.5, 8.5, 7.5, 7.0,
                           6.5, 6.0, 5.5, 5.0, 4.5], index=dates)
        vol = pd.Series([0.10] * 10, index=dates)
        t_events = pd.DatetimeIndex([dates[0]])

        # pt_sl=[1,2] → 止盈=10%, 止损=20%
        result = apply_triple_barrier(close, t_events, pt_sl=[1, 2],
                                       vol=vol, num_days=20, detect_limits=False)

        assert result.loc[dates[0], 'bin'] == -1   # 触止损
        assert result.loc[dates[0], 'ret'] < 0     # 负收益

    def test_hits_vertical_barrier(self):
        """价格未碰水平屏障 → 时间到期 → 标签 0"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 价格小幅波动，不碰任何屏障
        close = pd.Series([10.0, 10.1, 10.2, 10.1, 10.0,
                           9.9, 10.1, 10.2, 10.1, 10.0], index=dates)
        vol = pd.Series([0.10] * 10, index=dates)  # 屏障 ±10~20%
        t_events = pd.DatetimeIndex([dates[0]])

        # pt_sl=[2,2] → ±20%，价格波动 < 屏障
        result = apply_triple_barrier(close, t_events, pt_sl=[2, 2],
                                       vol=vol, num_days=8, detect_limits=False)

        assert result.loc[dates[0], 'bin'] == 0    # 时间到期

    def test_upper_barrier_before_lower(self):
        """上屏障和下屏障都触达，取先触达的"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        # 先跌后涨，但都没碰屏障
        close = pd.Series([10.0, 9.0, 11.0, 11.5, 12.0,
                           12.5, 13.0, 13.5, 14.0, 14.5], index=dates)
        vol = pd.Series([0.10] * 10, index=dates)
        t_events = pd.DatetimeIndex([dates[0]])

        # pt_sl=[2,1] → 止盈=20%, 止损=10%
        # 第2天跌到9.0（-10%）= 触发止损
        # 但不会涨到12.0（+20%）
        result = apply_triple_barrier(close, t_events, pt_sl=[2, 1],
                                       vol=vol, num_days=20, detect_limits=False)

        assert result.loc[dates[0], 'bin'] == -1   # 止损先触达

    def test_no_vol_zero(self):
        """波动率为负或零时，使用默认值 0.01（不会除零）"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        close = pd.Series([10.0] * 5, index=dates)
        vol = pd.Series([0.0, 0.0, -0.01, 0.0, 0.0], index=dates)
        t_events = pd.DatetimeIndex([dates[0]])

        # 不应崩溃
        result = apply_triple_barrier(close, t_events, pt_sl=[2, 1],
                                       vol=vol, num_days=5, detect_limits=False)

        assert len(result) == 1  # 应该正常返回一个结果

    def test_empty_events(self):
        """无事件 → 空 DataFrame"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        close = pd.Series(np.arange(10, 15), index=dates)
        vol = pd.Series([0.10] * 5, index=dates)
        t_events = pd.DatetimeIndex([])

        result = apply_triple_barrier(close, t_events, pt_sl=[2, 1],
                                       vol=vol, num_days=5)

        assert len(result) == 0


class TestMetaLabels:
    """Meta-Labeling 标签测试"""

    def test_meta_label_correct(self):
        """主模型方向 == 实际方向 → meta_label = 1"""
        from labeling.triple_barrier import get_meta_labels

        # 构造 TB 结果：获利 +5%
        tb_result = pd.DataFrame({
            't1': pd.date_range('2024-01-03', periods=2, freq='B'),
            'ret': [5.0, -3.0],      # 第一个赚, 第二个亏
        }, index=pd.DatetimeIndex(['2024-01-02', '2024-01-05']))

        side = pd.Series([1, -1], index=tb_result.index)  # 做多, 做空

        meta = get_meta_labels(tb_result, side)

        assert meta.loc[tb_result.index[0], 'bin'] == 1   # 做多 + 赚了 → 可执行
        assert meta.loc[tb_result.index[1], 'bin'] == 1   # 做空 + 亏了 (方向相反) → wait...

    def test_meta_label_skip(self):
        """主模型方向错误 → meta_label = 0"""
        from labeling.triple_barrier import get_meta_labels

        tb_result = pd.DataFrame({
            't1': pd.date_range('2024-01-03', periods=2, freq='B'),
            'ret': [-5.0, 3.0],
        }, index=pd.DatetimeIndex(['2024-01-02', '2024-01-05']))

        side = pd.Series([1, -1], index=tb_result.index)  # 做多, 做空

        meta = get_meta_labels(tb_result, side)

        assert meta.loc[tb_result.index[0], 'bin'] == 0   # 做多但亏了 → 跳过
        assert meta.loc[tb_result.index[1], 'bin'] == 0   # 做空但赚了 → 跳过
