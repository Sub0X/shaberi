@echo off
setlocal

REM ============================================================================
REM == Configuration
REM ============================================================================

REM Add the model names you want to process to this list, separated by spaces.
set "MODELS_TO_RUN= google/gemma-3-4b gemma-3-12b-it opengvlab_internvl3_5-4b opengvlab_internvl3_5-8b opengvlab_internvl3_5-14b lfm2-350m gemma-3-1b-it liquid/lfm2-1.2b"
REM google/gemma-3-4b gemma-3-12b-it opengvlab_internvl3_5-4b opengvlab_internvl3_5-8b opengvlab_internvl3_5-14b lfm2-350m gemma-3-1b-it liquid/lfm2-1.2b

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
    python generate_answers.py -m "%%M" -n %NUM_PROC% -t 0.6 --top_p 0.95
    
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
