@echo off
REM GPU test of the refactored fast_tts package against PLAIN PyPI qwen-tts 0.1.1
REM (wheel extracted to .tmp_pypi_check; PYTHONPATH shadows the editable vendored install)
REM Runs on GPU index 1 only - GPU 0 is reserved for manual work.
set PYTHONPATH=G:\qwen-tts\.tmp_pypi_check
set CUDA_VISIBLE_DEVICES=1
G:\qwen-tts\.conda\python.exe "%~dp0test_v14_pypi.py" %*
