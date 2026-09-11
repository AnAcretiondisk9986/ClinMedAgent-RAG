@echo off
title ҽѧ�̲Ĺ���̨
chcp 936 >nul 2>nul
cd /d "%~dp0.."
setlocal

rem ���γ��� PATH ��� python / py -3 / ��Ŀ�Դ� .venv-ocr��Ҫ���� import fitz
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
    echo [����] û���ҵ����õ� Python ����������Ҫ Python 3.10+ ���Ѱ�װ PyMuPDF����
    echo        ��װ������python -m pip install PyMuPDF
    echo.
    pause
    exit /b 1
)

echo.
echo  ������������ҽѧ�̲Ĺ���̨...
echo  Ĭ�ϵ�ַ��http://127.0.0.1:17173/   ��������Զ���
echo  �رձ����ڼ�ֹͣ���񣻸Ķ˿ڿ����У�start_web.bat --port 18080
echo.

%PYTHON% -X utf8 -m medical_rag.webapp %*

if errorlevel 1 (
    echo.
    echo [����] ��վ����ʧ�ܣ���鿴�Ϸ���ʾ��
    pause
    exit /b 1
)
endlocal
