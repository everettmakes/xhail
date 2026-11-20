#!/usr/bin/env bash
# Retry/feedback experiments (MLX, Apple Silicon). Run from experiments/bias_llm with the repo .venv.
#   A: 7B instruct, 3-shot, k=4: single vs resample vs XHAIL feedback
#   B: 1.5B fine-tuned on gold (train_mlx.py), k=4: single vs resample
set -euo pipefail
PY=../../.venv/bin/python
BIG=mlx-community/Qwen2.5-7B-Instruct-4bit
SMALL=mlx-community/Qwen2.5-1.5B-Instruct-4bit
ADAPTER=checkpoints/mlx-qwen1.5b

# One model at a time, with a pause between runs and a wait while macOS reports thermal pressure.
cool() {
  sleep 30
  while pmset -g therm | grep -qiE "warning level: *[1-9]|CPU_Speed_Limit *= *[1-9][0-9]?$"; do
    echo "thermal warning, waiting"; sleep 60
  done
}
run() { cool; $PY refine.py "$@"; }

for split in benchmarks data/test.jsonl; do
  tag=$([ "$split" = benchmarks ] && echo bench || echo test)
  for s in single resample feedback; do
    run --model $BIG --fewshot 3 --tasks $split --strategy $s --k 4 --name "7b_${s}_${tag}"
  done
  for s in single resample; do
    run --model $SMALL --adapter $ADAPTER --tasks $split --strategy $s --k 4 --name "ft1.5b_${s}_${tag}"
  done
done
