#!/usr/bin/env bash
set -euo pipefail

SIZES=("3B" "7B")
VARIANT_SLUGS=("qkvo" "qkvo_gate")
VARIANT_DIRS=("QKVO" "QKVO-Gate")
CKPTS=("step1k" "step2k" "best" "final")
OUT_DIR="results_valdev"
GT_DATASET="hf:lewiswatson/Frame2KG-YC2:validation_dev"

mkdir -p "${OUT_DIR}"

# Iterate over each configuration and run the evaluation
for size in "${SIZES[@]}"; do
  for idx in "${!VARIANT_SLUGS[@]}"; do
    variant_slug="${VARIANT_SLUGS[idx]}"
    variant_dir="${VARIANT_DIRS[idx]}"

    for ckpt in "${CKPTS[@]}"; do
      pred_dir="preds/${size}/${variant_dir}/${ckpt}"
      if [[ ! -d "${pred_dir}" ]]; then
        echo "[WARN] Skipping missing predictions directory: ${pred_dir}" >&2
        continue
      fi

      out_file="${OUT_DIR}/${size}__${variant_slug}__${ckpt}__frozen.csv"
      echo "[INFO] Evaluating ${pred_dir} -> ${out_file}"

      frame2kg-eval \
        --pred-dir "${pred_dir}" \
        --gt "${GT_DATASET}" \
        --text-mode semantic \
        --text-fields label --text-fields attributes \
        --tau 0.3 --alpha 0.7 \
        --out "${out_file}"
    done
  done
done
