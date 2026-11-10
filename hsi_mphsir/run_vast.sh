#!/usr/bin/env bash
# MP-HSIR pilot on one rented GPU: Stage 0 reproduction, then control vs D1 (physics noise) fine-tunes with equal
# wall-clock budgets, then paired evaluation. Results are pushed to a private HF repo after every stage.
#
# Env: HF_TOKEN (required); HSI_DEADLINE (unix time to finish by, default now + 10 h); HSI_OUT (default
#      /workspace/hsi_run); HSI_DATA (default /workspace/hsi_data); HSI_REPO (default tqhuyen/hsi-mphsir-pilot);
#      HSI_SMOKE=1 (2-minute arms, for checking the pipeline end to end).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${HSI_OUT:-/workspace/hsi_run}"
DATA="${HSI_DATA:-/workspace/hsi_data}"
REPO="${HSI_REPO:-tqhuyen/hsi-mphsir-pilot}"
DEADLINE="${HSI_DEADLINE:-$(( $(date +%s) + 36000 ))}"
MPHSIR_COMMIT=ee1a6fd3a6a8f0ab679fb53da509d4b64a2af6b1
export MPHSIR_DIR="${MPHSIR_DIR:-/workspace/MP-HSIR}"
: "${HF_TOKEN:?missing HF_TOKEN}"
mkdir -p "$OUT/outputs"

upload() {
  python - "$OUT/outputs" "$REPO" "$1" <<'PY' || echo "[hsi] upload failed: $1"
import os, sys
from huggingface_hub import HfApi
folder, repo, message = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder, commit_message=message)
print(f"[hsi] uploaded {message} -> hf://{repo}", flush=True)
PY
}

if [ ! -d "$MPHSIR_DIR/.git" ]; then
  git clone -q https://github.com/ZhehuiWu/MP-HSIR.git "$MPHSIR_DIR"
fi
git -C "$MPHSIR_DIR" checkout -q "$MPHSIR_COMMIT"
python -m pip install -q einops "timm<1" ftfy regex tifffile h5py gdown pandas scipy huggingface_hub \
  "git+https://github.com/openai/CLIP.git"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$HERE"
python data.py --root "$DATA" 2>&1 | tee -a "$OUT/outputs/data.log" || { echo "[hsi] data preparation failed"; exit 1; }
cp "$DATA/train_sources.json" "$OUT/outputs/" 2>/dev/null || true

python evaluate.py --data "$DATA" --checkpoint official --out "$OUT/outputs/eval.csv" 2>&1 | tee -a "$OUT/outputs/eval.log" \
  || echo "[hsi] stage 0 evaluation failed"
upload "stage 0: official checkpoint evaluation"

eval_reserve=$(( 2 * 15 * 60 + 15 * 60 ))
remaining=$(( DEADLINE - $(date +%s) - eval_reserve ))
if [ "${HSI_SMOKE:-0}" = "1" ]; then arm_minutes=2; else arm_minutes=$(( remaining / 2 / 60 )); fi
echo "[hsi] $(( remaining / 60 )) min available -> $arm_minutes min per arm"
if [ "$arm_minutes" -lt 2 ]; then
  echo "[hsi] no time left for fine-tuning"
else
  for arm in control physics; do
    python train.py --arm "$arm" --minutes "$arm_minutes" --data "$DATA" --out "$OUT/$arm" 2>&1 \
      | tee -a "$OUT/outputs/train_$arm.log" || { echo "[hsi] $arm training failed"; continue; }
    cp "$OUT/$arm/train_log.jsonl" "$OUT/outputs/train_${arm}_log.jsonl"
    cp "$OUT/$arm/train_summary.json" "$OUT/outputs/train_${arm}_summary.json"
    python evaluate.py --data "$DATA" --checkpoint "$OUT/$arm/last.pt" --tag "$arm" --out "$OUT/outputs/eval.csv" 2>&1 \
      | tee -a "$OUT/outputs/eval.log" || echo "[hsi] $arm evaluation failed"
    mkdir -p "$OUT/outputs/checkpoints"
    cp "$OUT/$arm/last.pt" "$OUT/outputs/checkpoints/$arm.pt"
    upload "arm $arm: fine-tune + evaluation"
  done
fi

python - "$OUT/outputs" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
table = pd.read_csv(out / "eval.csv")
means = table.groupby(["condition", "checkpoint"])[["psnr", "ssim", "sam"]].mean().unstack("checkpoint").round(3)
means.to_csv(out / "summary_by_condition.csv")
per_set = table.groupby(["set", "condition", "checkpoint"]).psnr.mean().unstack("checkpoint").round(2)
per_set.to_csv(out / "summary_by_set.csv")
summary = {"psnr_by_condition": {c: means["psnr"].loc[c].dropna().to_dict() for c in means.index}}
gate = out / "stage0_gate.json"
if gate.exists():
    summary["stage0_gate"] = json.loads(gate.read_text())
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(means["psnr"].to_string())
PY
date -u +%FT%TZ > "$OUT/outputs/DONE"
upload "final summary"
