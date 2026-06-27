#!/bin/bash
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"

echo "=== Phase 1: tiny-2ag (80M) — 4 进程并行 ==="
caffeinate python main.py --algorithm iac         --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42 &
PID1=$!
caffeinate python main.py --algorithm snac        --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42 &
PID2=$!
caffeinate python main.py --algorithm seac        --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42 &
PID3=$!
caffeinate python main.py --algorithm seac_pooled --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo "=== tiny-2ag done ==="

echo "=== Phase 2: small-4ag (150M) — 4 进程并行 ==="
caffeinate python main.py --algorithm iac         --env rware-small-4ag-v2 --total-steps 150000000 --seed 42 &
caffeinate python main.py --algorithm snac        --env rware-small-4ag-v2 --total-steps 150000000 --seed 42 &
caffeinate python main.py --algorithm seac        --env rware-small-4ag-v2 --total-steps 150000000 --seed 42 &
caffeinate python main.py --algorithm seac_pooled --env rware-small-4ag-v2 --total-steps 150000000 --seed 42 &
wait
echo "=== All done ==="
