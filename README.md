# 供应链选股 v1.1

> Serenity 1.0 选股 + CUSUM 择时 + 追损 3.0σ 退出

---

## 快速开始

```bash
# 启动面板
cd ~/股票退出机制1.0
./run.sh app.py
# 浏览器打开 http://localhost:8501
```

## 每日流程

| 时间 | 操作 | 命令 |
|------|------|------|
| 盘中 | 启动面板 | `./run.sh app.py` |
| 盘中 | 看信号 → 手动买卖 | 面板 Tab 1 & 2 |
| 盘中 | 录持仓 → 跟踪止损 | 面板 Tab 3 |
| 收盘 | 更新选股 | `cd ~/.hermes/... && python3 pipeline.py` |

## 回测

```bash
# 用默认配置跑回测
./run.sh main.py --config configs/default.yaml --ts_code 000001.SZ

# 跑保守策略
./run.sh main.py --config configs/conservative.yaml --ts_code 000001.SZ

# 查看历史回测记录
./run.sh main.py --show-runs
```

## 测试

```bash
./run.sh pytest tests/ -v
```

## 项目文件

| 文件 | 用途 |
|------|------|
| `app.py` | **操作面板**（日常使用） |
| `main.py` | TB + Meta-Labeling + XGBoost 回测流水线 |
| `config_loader.py` | YAML 配置加载器 |
| `logging_config.py` | 统一日志配置（文件+终端） |
| `run_tracker.py` | 实验跟踪（自动记录每次回测结果） |
| `configs/` | YAML 策略配置文件 |
| `tests/` | pytest 单元测试（29 个） |
| `run.sh` | 启动脚本（自动处理 PYTHONPATH 污染） |
| `.github/workflows/test.yml` | GitHub Actions CI（自动跑测试） |

## 策略逻辑

详见 README 完整版。
