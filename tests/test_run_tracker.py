"""
实验跟踪器单元测试
"""
import os
import pytest

from run_tracker import save_run, load_runs


class TestRunTracker:
    """实验跟踪测试"""

    def test_save_and_load(self):
        """保存一条记录后能加载出来"""
        run_id = save_run(
            params={'cusum_vol_mult': 1.5, 'num_days': 20, 'meta_threshold': 0.3},
            metrics={'sharpe': 0.45, 'max_dd': -0.12, 'win_rate': 0.55},
            stock_code='000001.SZ',
            date_range='2024-01-01~2024-12-31',
        )

        assert run_id is not None
        assert len(run_id) > 5  # 有内容

        df = load_runs()
        assert len(df) >= 1
        # 应该能找到我们刚存的记录
        matches = df[df['run_id'] == run_id]
        assert len(matches) >= 1

    def test_multiple_runs(self):
        """保存多条记录，按时间降序"""
        save_run(
            params={'cusum_vol_mult': 1.0},
            metrics={'sharpe': 0.3},
            stock_code='000001.SZ',
        )
        save_run(
            params={'cusum_vol_mult': 2.0},
            metrics={'sharpe': 0.6},
            stock_code='000001.SZ',
        )

        df = load_runs()
        assert len(df) >= 2
        # 按时间降序
        timestamps = df['timestamp'].tolist()
        assert timestamps == sorted(timestamps, reverse=True)

    def test_same_params_same_id(self):
        """相同参数产生相同 run_id（去重友好）"""
        id1 = save_run(params={'cusum_vol_mult': 1.5}, metrics={'sharpe': 0.5}, stock_code='test')
        id2 = save_run(params={'cusum_vol_mult': 1.5}, metrics={'sharpe': 0.5}, stock_code='test')
        assert id1 == id2
