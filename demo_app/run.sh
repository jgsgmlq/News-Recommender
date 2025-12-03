#!/bin/bash

echo "============================================"
echo "  新闻推荐系统 Web Demo"
echo "  AI Guide Course Project"
echo "============================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python，请先安装Python 3.8+"
    exit 1
fi

echo "[1/3] 检查依赖..."
if ! python3 -c "import flask" &> /dev/null; then
    echo "[提示] 正在安装依赖..."
    pip3 install -r requirements.txt
fi

echo ""
echo "[2/3] 启动服务器..."
echo ""
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 退出"
echo ""

# 启动Flask应用
python3 app.py
