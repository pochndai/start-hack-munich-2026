#!/bin/bash
# Stress test of v12 next to v11 on one pinned logical CPU (run on the second Nebius VM, 22 Nov 2026).
# Part A: other setups. Part B: the official first run on a disturbed machine. Part C: under small memory caps.
# Every timed run is a fresh process of the version's own benchmark.py with the official --cpus 1.
PY=$HOME/shm/.venv/bin/python
SNAP=$HOME/shm/bench/.cache/snapshots
V11=$(ls -d $SNAP/f3f14542ea40* | head -1); V12=$(ls -d $SNAP/f4f5c40d53d8* | head -1)
OUT=$HOME/out/stress; rm -rf $OUT; mkdir -p $OUT/logs
CPU=15
echo "v11=$V11 v12=$V12 cpu=$CPU siblings=$(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"

echo "== A1 variations"; cd $HOME/shm && $PY tests/variations.py 2>&1 | grep -v findfont | tail -45
echo "== A2 test_host"; $PY tests/test_host.py 2>&1 | tail -4
echo "== A3 handwritten_jacobians"; $PY tests/handwritten_jacobians.py 2>&1 | tail -3
for i in 1 2 3 4; do echo "== A4 aligo v12 run $i"; (cd $V12 && taskset -c $CPU $PY -u benchmark_aligo.py --cpus 1 --output $OUT/aligo-v12-$i.json 2>&1 | grep "Complete first-run\|^PASS\|FAIL\|Error"); done

echo "condition,version,round,first_run_s,pass,exit" > $OUT/results.csv
one() { # condition version directory round [command prefix]
  local log=$OUT/logs/$1-$2-$4.log; shift 0
  ( cd $3 && ${@:5} taskset -c $CPU $PY -u benchmark.py --cpus 1 --repeats 3 --output $OUT/tmp.json ) > $log 2>&1; local rc=$?
  local t=$(grep -o "Complete first-run total (excluding validation): [0-9.]*" $log | grep -o "[0-9.]*$")
  echo "$1,$2,$4,$t,$(grep -c '^PASS' $log),$rc" | tee -a $OUT/results.csv
}
cond() { for r in $(seq 1 $2); do if (( r % 2 )); then one $1 v11 $V11 $r; one $1 v12 $V12 $r; else one $1 v12 $V12 $r; one $1 v11 $V11 $r; fi; done; }
stop() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; sleep 5; }
busy() { for c in $(seq 0 14); do taskset -c $c sh -c 'while :; do :; done' & done; }
churn() { for c in 0 1 2 3; do taskset -c $c $PY -c "
import numpy as np
while True:
    a = np.ones(1 << 28)  # 2 GiB, touched, then freed
    del a" & done; }

echo "== B quiet"; cond quiet 10
echo "== B others-busy (15 other logical CPUs at 100%)"; busy; sleep 2; cond others-busy 10; stop
echo "== B memory-churn (4 processes allocating and freeing 2 GiB)"; churn; sleep 3; cond memory-churn 10; stop
echo "== B busy-and-churn"; busy; churn; sleep 3; cond busy-and-churn 10; stop
echo "== B same-cpu (a busy loop on the benchmark's own CPU)"; taskset -c $CPU sh -c 'while :; do :; done' & sleep 1; cond same-cpu 6; stop

echo "== C memory caps (swap off)"
for cap in 1G 512M 384M 320M 256M; do for v in v11 v12; do
  d=$V11; [ $v = v12 ] && d=$V12
  one cap-$cap $v $d 1 systemd-run --user --wait --pipe -q -p MemoryMax=$cap -p MemorySwapMax=0 --working-directory=$d
done; done
echo "== DONE"
