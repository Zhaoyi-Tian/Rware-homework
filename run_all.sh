#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Phase 1: tiny-2ag (40M) — 3 进程并行 ==="
caffeinate python main.py --algorithm iac  --env rware-tiny-2ag-v2 --total-steps 40000000  --seed 42 &
PID1=$!
caffeinate python main.py --algorithm snac --env rware-tiny-2ag-v2 --total-steps 40000000  --seed 42 &
PID2=$!
caffeinate python main.py --algorithm seac --env rware-tiny-2ag-v2 --total-steps 40000000  --seed 42 &
PID3=$!
wait $PID1 $PID2 $PID3
echo "=== tiny-2ag done ==="

echo "=== Phase 2: small-4ag (75M) — 3 进程并行 ==="
caffeinate python main.py --algorithm iac  --env rware-small-4ag-v2 --total-steps 75000000 --seed 42 &
caffeinate python main.py --algorithm snac --env rware-small-4ag-v2 --total-steps 75000000 --seed 42 &
caffeinate python main.py --algorithm seac --env rware-small-4ag-v2 --total-steps 75000000 --seed 42 &
wait
echo "=== All done ==="
