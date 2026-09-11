@echo off
title 医学教材工作台
chcp 936 >nul 2>nul
cd /d "%~dp0"
setlocal

rem 依次尝试 PATH 里的 python / py -3 / 项目自带 .venv-ocr，要求能 import fitz
set "PYTHON="
where python >nul 2>nul && python -c "import fitz" >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    where py >nul 2>nul && py -3 -c "import fitz" >nul 2>nul && set "PYTHON=py -3"
)
if not defined PYTHON (
    if exist ".venv-ocr\Scripts\python.exe" (
        ".venv-ocr\Scripts\python.exe" -c "import fitz" >nul 2>nul && set "PYTHON=.venv-ocr\Scripts\python.exe"
    )
)
if not defined PYTHON (
    echo.
    echo [错误] 没有找到可用的 Python 解释器（需要 Python 3.10+ 且已安装 PyMuPDF）。
    echo        安装依赖：python -m pip install PyMuPDF
    echo.
    pause
    exit /b 1
)

echo.
echo  正在启动本地医学教材工作台...
echo  默认地址：http://127.0.0.1:17173/   浏览器会自动打开
echo  关闭本窗口即停止服务；改端口可运行：start_web.bat --port 18080
echo.

%PYTHON% -X utf8 -m medical_rag.webapp %*

if errorlevel 1 (
    echo.
    echo [错误] 网站启动失败，请查看上方提示。
    pause
    exit /b 1
)
endlocal
