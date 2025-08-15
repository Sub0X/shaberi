@echo off
REM Usage: rename_model_files.bat old_model_name new_model_name

setlocal enabledelayedexpansion

if "%~2"=="" (
    echo Usage: %~nx0 old_model_name new_model_name
    exit /b 1
)

set "OLD=%~1"
set "NEW=%~2"

REM Recursively find files containing the old model name and rename them
for /r %%F in (*%OLD%*) do (
    set "FILE=%%~nxF"
    set "DIR=%%~dpF"
    set "NEWFILE=!FILE:%OLD%=%NEW%!"
    if not "!FILE!"=="!NEWFILE!" (
        echo Renaming "%%F" to "!DIR!!NEWFILE!"
        ren "%%F" "!NEWFILE!"
    )
)

echo Done.
