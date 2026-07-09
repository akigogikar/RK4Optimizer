#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /Users/akhileshgogikar/RK4Optimizer
echo "[EXP1] multi-seed n=10 start $(date +%T)"
python3 fullbatch_experiment.py --budget 600 --seeds 0 1 2 3 4 5 6 7 8 9 > exp1_multiseed.log 2>&1
cp fullbatch_results.json fullbatch_multiseed_results.json
echo "[EXP1] done $(date +%T)"
echo "[EXP2] h0 sweep start $(date +%T)"
python3 fullbatch_experiment.py --budget 600 --seeds 0 1 2 --lrs 0.001 0.0015 0.002 0.0025 0.003 > exp2_h0sweep.log 2>&1
cp fullbatch_results.json fullbatch_h0sweep_results.json
echo "[EXP2] done $(date +%T)"
echo "ALL_DONE"
