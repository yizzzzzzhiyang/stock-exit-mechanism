#!/bin/bash
# 启动脚本 — 自动处理 PYTHONPATH 污染问题
#
# Hermes Agent 设了 PYTHONPATH（Python 3.11），会污染本项目 venv（3.9）。
# 此脚本在启动前清除它。
#
# 用法:
#   ./run.sh main.py --config configs/default.yaml --ts_code 000001.SZ
#   ./run.sh pytest tests/ -v
#   ./run.sh -c "from config_loader import load_config; print(load_config())"

set -e

# 项目根目录
DIR="$(cd "$(dirname "$0")" && pwd)"

# 清除 PYTHONPATH（防止被 Hermes 的 3.11 包污染）
export PYTHONPATH=""

# 使用 venv 的 Python
PYTHON="$DIR/venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "错误：未找到 venv Python: $PYTHON"
    echo "请先创建虚拟环境: python3 -m venv venv"
    exit 1
fi

cd "$DIR"
exec "$PYTHON" "$@"
