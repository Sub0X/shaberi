@echo off
setlocal

REM ============================================================================
REM == Configuration
REM ============================================================================

REM Add the model names you want to process to this list, separated by spaces.
set "MODELS_TO_RUN=mistral-small-3.1-24b-instruct-2503 mistral-small-3.2-24b-instruct-2506 aya-expanse-32b-abliterated amoral-gemma3-27b-v2-qat"

REM Set the judge model to use for the evaluation.
set "JUDGE_MODEL=gpt-4.1"

REM Set the number of processes to use for the map function in judge_answers.py.
set "NUM_PROC=8"

REM Set the dataset to use for the evaluation.
set "DATASET=lmg-anon/VNTL-v3.1-1k"

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
    python judge_answers.py -m "%%M" -e %JUDGE_MODEL% -n %NUM_PROC% -d %DATASET%
    
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
