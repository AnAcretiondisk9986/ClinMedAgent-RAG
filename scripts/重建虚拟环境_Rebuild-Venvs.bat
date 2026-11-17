@echo off
chcp 936 >nul 2>nul
title Rebuild-Venvs 重建虚拟环境
setlocal EnableExtensions
cd /d "%~dp0.."
set "CORE=依赖-核心_requirements-core.txt"
set "OCR=依赖-OCR识别_requirements-ocr.txt"
set "PADDLE=依赖-版面表格_requirements-paddle.txt"

echo.
echo  重建虚拟环境：换电脑、换 Windows 用户，或把项目目录移动到别处后使用。
echo  Rebuild virtualenvs after moving the project folder or changing machine.
echo  项目目录: %CD%
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo [错误] 找不到 Windows py 启动器。请安装 Python 3.12 与 3.14 后重试。
  echo        手动方式: python -m venv .venv-ocr  ^&^&  .venv-ocr\Scripts\python.exe -m pip install -r "%CORE%" -r "%OCR%"
  pause
  exit /b 1
)

set "FAILED=0"

echo [1/3] .venv-ocr （Python 3.14，OCR / 纠错 / 结构化）
if exist ".venv-ocr\Scripts\python.exe" (
  ".venv-ocr\Scripts\python.exe" -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    echo        已可运行，跳过
    goto ocr_done
  )
  echo        现有环境已失效，先删除再重建
  rmdir /s /q ".venv-ocr"
)
py -3.14 -m venv .venv-ocr
if errorlevel 1 py -3 -m venv .venv-ocr
if not exist ".venv-ocr\Scripts\python.exe" (
  echo [错误] 创建 .venv-ocr 失败：请确认已安装 Python 3.14。
  set "FAILED=1"
  goto ocr_done
)
".venv-ocr\Scripts\python.exe" -m pip install --upgrade pip
".venv-ocr\Scripts\python.exe" -m pip install -r "%CORE%" -r "%OCR%"
if errorlevel 1 set "FAILED=1"
:ocr_done

echo [2/3] .venv-ocr312 （Python 3.12，版面 / 表格）
if exist ".venv-ocr312\Scripts\python.exe" (
  ".venv-ocr312\Scripts\python.exe" -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    echo        已可运行，跳过
    goto pad_done
  )
  echo        现有环境已失效，先删除再重建
  rmdir /s /q ".venv-ocr312"
)
py -3.12 -m venv .venv-ocr312
if not exist ".venv-ocr312\Scripts\python.exe" (
  echo [错误] 创建 .venv-ocr312 失败：请确认已安装 Python 3.12。
  set "FAILED=1"
  goto pad_done
)
".venv-ocr312\Scripts\python.exe" -m pip install --upgrade pip
".venv-ocr312\Scripts\python.exe" -m pip install -r "%CORE%" -r "%PADDLE%"
if errorlevel 1 set "FAILED=1"
:pad_done

echo [3/3] 结果
if "%FAILED%"=="1" (
  echo [警告] 有步骤失败。请按 README 的「安装 OCR 环境 / 安装 Paddle 环境」手动处理。
) else (
  echo 完成。验证方式: python -m medical_rag.cli doctor
)
echo.
pause
endlocal