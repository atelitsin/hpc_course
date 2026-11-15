#!/bin/bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/bench \
    --out results/benchmark.csv \
    --sizes 1000,2000,4000,8000 \
    --threads 1,2,4,6,8 \
    --chunks 1,16,64 \
    --reps 7
