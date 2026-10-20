#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot end-to-end run with the real DeepSeek API.
#
#   ./run_e2e_test.sh              # live, writes outputs/e2e_result.html once
#   ./run_e2e_test.sh --mode mock  # offline preview, zero cost
#
# The script refuses to overwrite an existing live report unless --force is
# supplied, so debugging cannot silently burn API credits.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-analytics-agent}"
CONDA_BASE="$(conda info --base)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BASE}/envs/${ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] 未找到 conda 环境 ${ENV_NAME} 的 Python：${PYTHON_BIN}" >&2
  echo "        请先执行: conda env create -f environment.yml" >&2
  exit 1
fi

if [[ ! -f environment.env ]]; then
  echo "[error] 缺少 environment.env，请复制 environment.env.example 并填入 DEEPSEEK_API_KEY" >&2
  exit 1
fi

echo "[env] ${PYTHON_BIN}"
exec "${PYTHON_BIN}" scripts/run_e2e_test.py "$@"
