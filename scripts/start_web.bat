@echo off
title 医学教材工作台
chcp 936 >nul 2>nul
cd /d "%~dp0.."
setlocal

rem 按顺序探测可用解释器：PATH 里的 python / py -3，再到项目自带的虚拟环境。
rem 启动网页需要能 import fitz（PyMuPDF），探测不到就给出中文提示。
set "PYTHON="
where python >nul 2>nul && python -c "import fitz" >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    where py >nul 2>nul && py -3 -c "import fitz" >nul 2>nul && set "PYTHON=py -3"
)
if not defined PYTHON (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -c "import fitz" >nul 2>nul && set "PYTHON=.venv\Scripts\python.exe"
    )
)
if not defined PYTHON (
    if exist ".venv-ocr\Scripts\python.exe" (
        ".venv-ocr\Scripts\python.exe" -c "import fitz" >nul 2>nul && set "PYTHON=.venv-ocr\Scripts\python.exe"
    )
)
if not defined PYTHON (
    if exist ".venv-ocr312\Scripts\python.exe" (
        ".venv-ocr312\Scripts\python.exe" -c "import fitz" >nul 2>nul && set "PYTHON=.venv-ocr312\Scripts\python.exe"
    )
)
if not defined PYTHON (
    echo.
    echo [错误] 没有找到可用的 Python 解释器（需要 Python 3.10+，且已安装 PyMuPDF）。
    echo        安装命令：python -m pip install PyMuPDF
    echo        也可以先激活项目自带的虚拟环境，再运行本脚本。
    echo.
    pause
    exit /b 1
)

echo.
echo  正在启动本地医学教材工作台...
echo  默认地址：http://127.0.0.1:17173/   （启动后会自动打开浏览器）
echo  关闭本窗口即停止服务；改端口可以传参：start_web.bat --port 18080
echo.

%PYTHON% -X utf8 -m medical_rag.webapp %*

if errorlevel 1 (
    echo.
    echo [错误] 网站启动失败，请查看上方提示信息。
    pause
    exit /b 1
)
endlocal
