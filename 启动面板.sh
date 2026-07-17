#!/bin/bash
# ================================================================
#  股票退出机制 v1.1 · 操作面板 · 一键启动
# ================================================================
set -e

PROJECT_DIR="$HOME/股票退出机制1.0"
cd "$PROJECT_DIR"

echo "========================================"
echo "  股票退出机制 v1.1 · 操作面板"
echo "========================================"

# 检查 Python
PYTHON=/usr/bin/python3
if [ ! -f "$PYTHON" ]; then
    echo "❌ 未找到 /usr/bin/python3"
    exit 1
fi

# 检查 NAS
if [ -d "/Volumes/quant/stocks" ]; then
    echo "✅ NAS 已挂载"
else
    echo "⚠️  NAS 未挂载，将使用 tushare 数据"
fi

# 启动
echo "🚀 启动面板..."
echo "   浏览器打开: http://localhost:8501"
echo "   停止: Ctrl+C"
echo ""

$PYTHON -m streamlit run app.py --server.port 8501
