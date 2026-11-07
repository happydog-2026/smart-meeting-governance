@echo off
rem 旧版 Streamlit 界面(备用入口)。新版界面直接双击 start_web.bat 即可。
cd /d "%~dp0"
set "MEETNOTE_UI=streamlit"
call start_web.bat
