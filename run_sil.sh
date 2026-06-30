#!/bin/bash
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== SIL 收敛速率对比: small-4ag 50M — 4 进程并行 ==="
caffeinate python main.py --algorithm iac      --env rware-small-4ag-v2 --total-steps 50000000 --seed 42 --log-dir ./logs/sil --checkpoint-dir ./checkpoints/sil &
PID1=$!
caffeinate python main.py --algorithm seac     --env rware-small-4ag-v2 --total-steps 50000000 --seed 42 --log-dir ./logs/sil --checkpoint-dir ./checkpoints/sil &
PID2=$!
caffeinate python main.py --algorithm iac_sil  --env rware-small-4ag-v2 --total-steps 50000000 --seed 42 --log-dir ./logs/sil --checkpoint-dir ./checkpoints/sil &
PID3=$!
caffeinate python main.py --algorithm seac_sil --env rware-small-4ag-v2 --total-steps 50000000 --seed 42 --log-dir ./logs/sil --checkpoint-dir ./checkpoints/sil &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo "=== All done ==="
