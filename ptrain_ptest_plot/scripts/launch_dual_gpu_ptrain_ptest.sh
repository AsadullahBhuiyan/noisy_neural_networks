#!/usr/bin/env bash
set -euo pipefail

# Dual-GPU launcher for the fully-corrupted train p_train / fixed-corrupted test p_test MNIST grid.
# Usage:
#   bash launch_dual_gpu_ptrain_ptest.sh [extra train_noisy_mnist_ptrain_ptest.py args...]

ENV_NAME="${ENV_NAME:-pytorch}"
PY_SCRIPT="${PY_SCRIPT:-train_noisy_mnist_ptrain_ptest.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_noisy_mnist_ptrain_ptest.py}"
RESULT_DIR="${RESULT_DIR:-./results/ptrain_ptest}"
TAG="${TAG:-dual_gpu_ptrain_ptest_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
USE_CUDA="${USE_CUDA:-1}"

mkdir -p "${RESULT_DIR}"

RAW0="${RESULT_DIR}/${TAG}_gpu${GPU0}.csv"
RAW1="${RESULT_DIR}/${TAG}_gpu${GPU1}.csv"
LOG0="${RESULT_DIR}/${TAG}_gpu${GPU0}.log"
LOG1="${RESULT_DIR}/${TAG}_gpu${GPU1}.log"
MERGED="${RESULT_DIR}/${TAG}_merged.csv"
PNG="${RESULT_DIR}/${TAG}_accuracy.png"
PDF="${RESULT_DIR}/${TAG}_accuracy.pdf"

CUDA_ARG=()
if [[ "${USE_CUDA}" == "1" ]]; then
  CUDA_ARG=(--use-cuda)
fi

echo "Launching p_train shard 0 on GPU ${GPU0}"
CUDA_VISIBLE_DEVICES="${GPU0}" \
conda run -n "${ENV_NAME}" python "${PY_SCRIPT}" \
  "$@" \
  "${CUDA_ARG[@]}" \
  --run-offset 0 \
  --run-stride 2 \
  --num-workers "${NUM_WORKERS}" \
  --results-path "${RAW0}" \
  > "${LOG0}" 2>&1 &
PID0=$!

echo "Launching p_train shard 1 on GPU ${GPU1}"
CUDA_VISIBLE_DEVICES="${GPU1}" \
conda run -n "${ENV_NAME}" python "${PY_SCRIPT}" \
  "$@" \
  "${CUDA_ARG[@]}" \
  --run-offset 1 \
  --run-stride 2 \
  --num-workers "${NUM_WORKERS}" \
  --results-path "${RAW1}" \
  > "${LOG1}" 2>&1 &
PID1=$!

echo "Waiting for both shards..."
wait "${PID0}"
wait "${PID1}"
echo "Both shards completed."

head -n 1 "${RAW0}" > "${MERGED}"
tail -n +2 "${RAW0}" >> "${MERGED}"
tail -n +2 "${RAW1}" >> "${MERGED}"

conda run -n "${ENV_NAME}" python "${PLOT_SCRIPT}" \
  --csv "${MERGED}" \
  --output-png "${PNG}" \
  --output-pdf "${PDF}"

echo "Done."
echo "Logs:"
echo "  ${LOG0}"
echo "  ${LOG1}"
echo "Outputs:"
echo "  ${RAW0}"
echo "  ${RAW1}"
echo "  ${MERGED}"
echo "  ${PNG}"
echo "  ${PDF}"
