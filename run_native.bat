@echo off
REM Run native Qwen3TTSModel streaming test
set PYTHON=G:\qwen-tts\.conda\python.exe
%PYTHON% "%~dp0run_native.py" %*
