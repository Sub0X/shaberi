@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Rename Script for Model Identifiers
REM ============================================================================
REM This script recursively renames files and directories in the data folder
REM that contain the OLD_IDENTIFIER to NEW_IDENTIFIER

REM Configuration - Set these variables to specify what to rename
set "OLD_IDENTIFIER=liquid__"
set "NEW_IDENTIFIER="

REM Determine the correct data directory path based on current location
if exist "data" (
    set "DATA_DIR=data"
) else if exist "..\data" (
    set "DATA_DIR=..\data"
) else (
    echo ERROR: Could not find data directory. Please run this script from the project root or scripts directory.
    pause
    exit /b 1
)

echo Starting rename operation in %DATA_DIR%
echo Old identifier: %OLD_IDENTIFIER%
echo New identifier: %NEW_IDENTIFIER%
echo.

REM Rename directories first (to avoid conflicts)
echo Renaming directories...
for /d /r "%DATA_DIR%" %%D in (*) do (
    if "%%~nxD" neq "" (
        set "DIRNAME=%%~nxD"
        set "NEWDIRNAME=!DIRNAME:%OLD_IDENTIFIER%=%NEW_IDENTIFIER%!"
        if not "!DIRNAME!"=="!NEWDIRNAME!" (
            echo Renaming directory: %%D
            echo   To: %%~dpD!NEWDIRNAME!
            ren "%%D" "!NEWDIRNAME!"
        )
    )
)

echo.
echo Renaming files...
for /r "%DATA_DIR%" %%F in (*) do (
    if "%%~nxF" neq "" (
        set "FILENAME=%%~nxF"
        set "NEWNAME=!FILENAME:%OLD_IDENTIFIER%=%NEW_IDENTIFIER%!"
        if not "!FILENAME!"=="!NEWNAME!" (
            echo Renaming file: %%F
            echo   To: %%~dpF!NEWNAME!
            ren "%%F" "!NEWNAME!"
        )
    )
)

echo.
echo Rename operation completed.
echo.
pause
