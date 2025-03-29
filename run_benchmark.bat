@echo off
echo Starting path planning benchmarks...

python benchmark.py --num_runs 3 --output benchmark_results.csv

if %ERRORLEVEL% EQU 0 (
    echo Benchmark completed successfully. Generating visualizations...
    python benchmark_visualization.py --input benchmark_results.csv --output_prefix viz_benchmark
    echo All done! Check benchmark_results.csv and viz_benchmark_*.png for results.
) else (
    echo Benchmark failed. Check the error messages above.
    exit /b 1
)