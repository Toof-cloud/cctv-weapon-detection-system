@echo off
setlocal
cd /d "%~dp0"

set "MOCKUP_PYTHON=%~dp0mockup_ui\.venv\Scripts\python.exe"

if not exist "%MOCKUP_PYTHON%" (
    echo The mockup UI virtual environment was not found.
    echo Expected: "%MOCKUP_PYTHON%"
    echo.
    echo Create it with:
    echo   py -3.13 -m venv mockup_ui\.venv
    echo   mockup_ui\.venv\Scripts\python.exe -m pip install -r mockup_ui\requirements-ui.txt
    pause
    exit /b 1
)

"%MOCKUP_PYTHON%" -B "%~dp0mockup_ui\app.py" %*
set "MOCKUP_EXIT_CODE=%ERRORLEVEL%"

if not "%MOCKUP_EXIT_CODE%"=="0" (
    echo.
    echo The mockup UI exited with code %MOCKUP_EXIT_CODE%.
    pause
)

exit /b %MOCKUP_EXIT_CODE%
