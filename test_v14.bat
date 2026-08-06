@echo off
REM TTS v14 streaming test suite (10 sentences)
set PYTHON=G:\qwen-tts\.conda\python.exe
%PYTHON% "%~dp0test_v14.py" %*
