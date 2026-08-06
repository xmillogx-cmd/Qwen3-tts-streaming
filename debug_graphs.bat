@echo off
REM CUDA graph timing debug
set PYTHON=G:\qwen-tts\.conda\python.exe
%PYTHON% "%~dp0debug_graphs.py" %*
