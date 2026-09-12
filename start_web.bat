@echo off
title 医学教材工作台
chcp 936 >nul 2>nul
setlocal

rem 根目录快捷入口：真正的启动逻辑在 scripts\start_web.bat，双击本文件即可。
if not exist "%~dp0scripts\start_web.bat" (
    echo.
    echo [错误] 找不到 scripts\start_web.bat，请确认项目文件完整。
    echo.
    pause
    exit /b 1
)

call "%~dp0scripts\start_web.bat" %*
exit /b %errorlevel%
