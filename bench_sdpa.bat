@echo off
REM SDPA attention benchmark
set PYTHON=G:\qwen-tts\.conda\python.exe
%PYTHON% "%~dp0bench_sdpa.py" %*
