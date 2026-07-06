@echo off
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo The uv command is required but was not found.
  pause
  exit /b 1
)
uv run --with "huggingface_hub>=0.34" python scripts\publish_huggingface_space.py
pause
