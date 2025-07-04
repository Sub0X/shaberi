@echo off
setlocal

REM ============================================================================
REM == Configuration
REM ============================================================================

REM Add the model names you want to process to this list, separated by spaces.
set "MODELS_TO_RUN=mistral-small-3.1-24b-instruct-2503 mistral-small-3.2-24b-instruct-2506 aya-expanse-32b-abliterated amoral-gemma3-27b-v2-qat"

REM Set the number of processes to use for the map function in generate_answers.py.
set "NUM_PROC=1"


REM ============================================================================
REM == Script Execution
REM ============================================================================
echo Starting benchmark answer generation for the following models:
echo %MODELS_TO_RUN%
echo.

REM Loop through each model in the list
for %%M in (%MODELS_TO_RUN%) do (
    echo =================================================================
    echo      Processing model: %%M
    echo =================================================================
    
    REM Run the python script to generate answers for the current model
    python generate_answers.py -m "%%M" -n %NUM_PROC% -d "lmg-anon/VNTL-v3.1-1k" 
    
    REM Check for errors after each command
    if %errorlevel% neq 0 (
        echo.
        echo ***************************************************
        echo * ERROR: An error occurred while processing %%M. *
        echo * Script will now terminate.                      *
        echo ***************************************************
        echo.
        goto :eof
    )
    
    echo.
    echo Finished processing model: %%M
    echo.
)

echo All models have been processed successfully.
endlocal
