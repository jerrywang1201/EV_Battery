#!/usr/bin/env bash
set -euo pipefail

RAW_CSV="${1:-data/raw/dataset.csv}"
FEATURES="${2:-Voltage,Current,Temperature}"
LABEL="${3:-SoH}"
GROUP="${4:-CellID}"
WINDOW="${5:-128}"
STRIDE="${6:-16}"
SEED="${7:-42}"
LABEL_FRAC="${8:-0.1}"

OUT_PROCESSED="data/processed"
OUT_SSL="runs/ssl"
OUT_DOWNSTREAM="runs/downstream"
OUT_TUNE="runs/tune"

python3 -m src.inspect --data "${RAW_CSV}"

python3 -m src.preprocess \
  --data "${RAW_CSV}" \
  --out "${OUT_PROCESSED}" \
  --features "${FEATURES}" \
  --label "${LABEL}" \
  --group "${GROUP}" \
  --window "${WINDOW}" \
  --stride "${STRIDE}" \
  --seed "${SEED}"

python3 -m src.ssl_pretrain \
  --data "${OUT_PROCESSED}/train.npz" \
  --out "${OUT_SSL}" \
  --seed "${SEED}"

python3 -m src.downstream \
  --data "${OUT_PROCESSED}/train.npz" \
  --val "${OUT_PROCESSED}/val.npz" \
  --ckpt "${OUT_SSL}/encoder.pt" \
  --label_frac "${LABEL_FRAC}" \
  --seed "${SEED}" \
  --out "${OUT_DOWNSTREAM}" \
  --save_best \
  --plot \
  --curve \
  --save_pred

python3 -m src.tune \
  --data "${RAW_CSV}" \
  --ckpt "${OUT_SSL}/encoder.pt" \
  --features "${FEATURES}" \
  --label "${LABEL}" \
  --group "${GROUP}" \
  --windows "64,128,256" \
  --strides "8,16" \
  --label_modes "last,mean" \
  --label_fracs "0.05,0.1" \
  --out "${OUT_TUNE}"
