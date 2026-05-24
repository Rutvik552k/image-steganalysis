#!/bin/bash
# Remote monitor - runs on GPU instance, logs every 5 min
for i in $(seq 1 20); do
  {
    echo "=== CHECK $i at $(date) ==="
    echo "---LOG TAIL---"
    tail -30 /tmp/training.log 2>&1
    echo "---LOG LINES---"
    wc -l /tmp/training.log 2>&1
    echo "---MEM---"
    free -h | head -2
    echo "---GPU---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1
    echo "---PROC---"
    ps aux | grep "train.py" | grep -v grep 2>&1
    echo ""
  } >> /tmp/monitor.log
  /bin/sleep 300
done
