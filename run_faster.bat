@echo off
REM Run FasterQwen3TTS (CUDA graphs) streaming test
set PYTHON=G:\qwen-tts\.conda\python.exe
%PYTHON% "%~dp0run_faster.py" %*
