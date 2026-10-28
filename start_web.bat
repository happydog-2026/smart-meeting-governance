@echo off
cd /d "%~dp0"

rem ---- pick the best available Python (prefer 3.13 / 3.12 / 3.11, then any) ----
set "PYCMD="
py -3.13 --version >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD py -3.12 --version >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD py -3.11 --version >nul 2>nul && set "PYCMD=py -3.11"
if not defined PYCMD py -3 --version >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD python --version >nul 2>nul && set "PYCMD=python"

if not defined PYCMD (
    echo Python not found. Please install Python 3.11-3.13:
    echo   https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

rem ---- run the cross-platform launcher (Chinese messages, auto install deps) ----
chcp 65001 >nul
%PYCMD% run.py
pause
