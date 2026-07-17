"""
配置加载器单元测试

覆盖场景：
  - 默认配置（无文件）
  - 加载 YAML 文件
  - 文件不存在抛 FileNotFoundError
  - 多路径搜索
"""
import os
import tempfile
import pytest

from config_loader import load_config


class TestLoadConfig:
    """配置加载测试"""

    def test_default_config(self):
        """无参数 → 返回默认配置"""
        cfg = load_config()

        # 必要字段完整
        assert 'strategy' in cfg
        assert 'barriers' in cfg
        assert 'cost' in cfg

        # 默认值正确
        assert cfg['strategy']['cusum_vol_mult'] == 1.5
        assert cfg['strategy']['num_days'] == 20
        assert cfg['barriers'] == [[2, 1], [2, 2], [3, 1], [1, 1]]
        assert cfg['cost']['commission'] == 0.00025

    def test_load_default_yaml(self):
        """加载 configs/default.yaml"""
        cfg = load_config('configs/default.yaml')

        assert cfg['strategy']['cusum_vol_mult'] == 1.5
        assert cfg['barriers'] == [[2, 1], [2, 2], [3, 1], [1, 1]]

    def test_load_conservative_yaml(self):
        """加载 configs/conservative.yaml"""
        cfg = load_config('configs/conservative.yaml')

        # 保守配置应覆盖默认值
        assert cfg['strategy']['cusum_vol_mult'] == 2.0
        assert cfg['strategy']['num_days'] == 15
        assert cfg['strategy']['meta_threshold'] == 0.5
        assert cfg['barriers'] == [[1, 1], [2, 2]]

        # 未覆盖的字段保留默认值
        assert cfg['cost']['commission'] == 0.00025

    def test_override_partial_config(self):
        """YAML 只有部分字段时，其余保留默认值"""
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        tmp.write("strategy:\n  meta_threshold: 0.8\n")
        tmp.close()

        cfg = load_config(tmp.name)
        os.unlink(tmp.name)

        # 被覆盖的
        assert cfg['strategy']['meta_threshold'] == 0.8
        # 未覆盖的保留默认
        assert cfg['strategy']['cusum_vol_mult'] == 1.5
        assert cfg['strategy']['num_days'] == 20
        assert cfg['barriers'] == [[2, 1], [2, 2], [3, 1], [1, 1]]

    def test_file_not_found(self):
        """不存在的文件 → FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_file.yaml')

    def test_cost_defaults(self):
        """交易成本默认值"""
        cfg = load_config()
        assert cfg['cost'] == {
            'commission': 0.00025,
            'stamp_tax': 0.001,
            'slippage': 0.001,
        }
