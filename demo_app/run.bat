@echo off
echo ============================================
echo   新闻推荐系统 Web Demo
echo   AI Guide Course Project
echo ============================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt
)

echo.
echo [2/3] 启动服务器...
echo.
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 退出
echo.

REM 启动Flask应用
python app.py

pause
