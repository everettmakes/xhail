#!/usr/bin/env bash
# Copy what the Colab notebook needs into a separate Drive folder (not the Drive git checkout).
# Colab writes predictions back to preds/v2 there; pull them with: sync_to_drive.sh --pull
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DRIVE="$HOME/Library/CloudStorage/GoogleDrive-jeverer@gmail.com/My Drive/Research Engineering/refresh-xhail/xhail-colab"

if [ "${1:-}" = "--pull" ]; then
  mkdir -p "$REPO/experiments/bias_llm/preds/v2"
  rsync -a "$DRIVE/experiments/bias_llm/preds/v2/" "$REPO/experiments/bias_llm/preds/v2/"
  exit 0
fi

mkdir -p "$DRIVE/experiments"
rsync -a "$REPO/pyproject.toml" "$REPO/README.md" "$REPO/LICENSE" "$DRIVE/"
rsync -a --delete --exclude __pycache__ "$REPO/xhail/" "$DRIVE/xhail/"
rsync -a --delete "$REPO/experiments/benchmarks/" "$DRIVE/experiments/benchmarks/"
rsync -a --exclude __pycache__ --exclude checkpoints --exclude preds --exclude results \
  --exclude huggingface_tokenizers_cache --exclude unsloth_compiled_cache --exclude '*.log' \
  "$REPO/experiments/bias_llm/" "$DRIVE/experiments/bias_llm/"
echo "synced to $DRIVE"
