#!/usr/bin/env bash
# Finds out why the current version is sometimes several times slower on a busy Linux machine.
cd "$(dirname "$0")"; TOP=$PWD
PY=$(command -v python3.12 || command -v python3)
{
echo "== machine"; grep -m1 'model name' /proc/cpuinfo; echo "logical CPUs: $(nproc)"; free -g | sed -n 2p; uptime; uname -r; echo "code: $(cat COMMIT.txt)"
echo "== large-page settings"; for f in enabled defrag khugepaged/defrag shmem_enabled; do echo "$f: $(cat /sys/kernel/mm/transparent_hugepage/$f 2>/dev/null)"; done
grep -i "hugepages_total\|AnonHugePages\|MemFree\|MemAvailable" /proc/meminfo; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed 's/^/cgroup memory limit: /'; cat /sys/fs/cgroup/cpu.max 2>/dev/null | sed 's/^/cgroup cpu limit: /'
} 2>&1 | tee machine.txt
for old in ../feyer-run3/venv ../feyer-run2/venv ../feyer-run/venv; do [ -d venv ] || { [ -x "$old/bin/python" ] && ln -sfn "$(cd "$old" && pwd -P)" venv; }; done
[ -d venv ] || $PY -m venv venv 2>/dev/null || { $PY -m pip install -q virtualenv && $PY -m virtualenv -q venv; }
. venv/bin/activate || { echo "could not create a Python environment"; exit 1; }
echo "== installing packages"
python -m pip install -q -r repo/requirements.txt >pip.log 2>&1 || { echo "package install failed, see pip.log"; exit 1; }
python -m pip install -q --no-deps -e repo >>pip.log 2>&1
rm -rf results; echo "== measuring (about 12 minutes)"
python diagnose.py 8 2>&1 | tee diagnose.log
tar czf results4.tar.gz machine.txt diagnose.log results/runs.jsonl COMMIT.txt 2>/dev/null
echo "== done ($(date +%H:%M:%S)): download results4.tar.gz from the file list on the left, then delete the instance"
