@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration
REM ============================================================================

REM Add the model names you want to process to this list, separated by spaces.
set "MODELS_TO_RUN="

REM When set to true the script will scan data\model_answers and judge all models found there.
set "judge_all=true"

REM Set the judge model to use for the evaluation.
set "JUDGE_MODEL=gpt-5-mini"

REM Set the number of processes to use for the map function in judge_answers.py.
set "NUM_PROC=4"

REM Set the temperature for the judge model.
set "TEMPERATURE=1.0"

REM Set the dataset to use for the evaluation.
set "DATASET="

REM ============================================================================
if /I "%judge_all%"=="true" (
    echo judge_all=true — scanning data\model_answers for model files...
    set "MODELS_TO_RUN="
    for /F "usebackq delims=" %%F in (`dir /B /S "data\model_answers\*.json"`) do (
        set "fname=%%~nF"
        if defined MODELS_TO_RUN (
            set "MODELS_TO_RUN=!MODELS_TO_RUN! !fname!"
        ) else (
            set "MODELS_TO_RUN=!fname!"
        )
    )
    echo Models discovered: !MODELS_TO_RUN!
)

REM ============================================================================
REM == Script Execution
REM ============================================================================
echo Starting benchmark judging for the following models:
echo %MODELS_TO_RUN%
echo Using judge model: %JUDGE_MODEL%
echo.

REM Loop through each model in the list
for %%M in (%MODELS_TO_RUN%) do (
    echo =================================================================
    echo      Judging model: %%M
    echo =================================================================
    
    REM Run the python script to judge the answers for the current model
    python judge_answers.py -m "%%M" -e %JUDGE_MODEL% -n %NUM_PROC% -t %TEMPERATURE%
    
    REM Check for errors after each command
    if %errorlevel% neq 0 (
        echo.
        echo ***************************************************
        echo * ERROR: An error occurred while judging %%M. *
        echo * Script will now terminate.                      *
        echo ***************************************************
        echo.
        goto :eof
    )
    
    echo.
    echo Finished judging model: %%M
    echo.
)

echo All models have been judged successfully.
endlocal
