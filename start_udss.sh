#!/bin/bash
# UDSS 启动脚本 - 启动后端API和前端开发服务器

echo "=================================="
echo "UDSS - 通用决策支持系统"
echo "=================================="
echo ""

# 启动后端API (后台运行)
echo "[1/2] 启动后端API (port 5000)..."
python udss_api.py &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端
echo "[2/2] 启动前端 (port 5173)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=================================="
echo "服务已启动:"
echo "  - UDSS API:    http://localhost:5000"
echo "  - Frontend:    http://localhost:5173"
echo "=================================="
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待信号
wait