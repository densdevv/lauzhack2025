@echo off
setlocal

REM Root wrapper: forwards to lauzhack\start_app.bat from any folder name (ZIP extract safe)
pushd "%~dp0" 1>nul 2>nul
if exist ".\lauzhack\start_app.bat" (
  call ".\lauzhack\start_app.bat"
  set "EXITCODE=%errorlevel%"
  if not "%EXITCODE%"=="0" (
    echo.
    echo [WRAPPER] The inner launcher returned exit code %EXITCODE%.
    echo Press any key to view the messages above and exit.
    pause >nul
    popd 1>nul 2>nul
    endlocal & exit /b %EXITCODE%
  )
  popd 1>nul 2>nul
  endlocal & exit /b 0
) else (
  echo [ERROR] Could not find lauzhack\start_app.bat next to this script.
  echo Ensure you extracted the ZIP preserving the folder structure.
  echo Current directory: %CD%
  echo Script location: %~dp0
  echo Press any key to exit.
  pause >nul
  popd 1>nul 2>nul
  endlocal & exit /b 1
)
