#!/usr/bin/env bash

set -euo pipefail

KERNEL_NAME="trading_tda"
DISPLAY_NAME="Python (trading-tda)"

echo "Installing ipykernel: ${DISPLAY_NAME}"

uv run python -m ipykernel install \
  --user \
  --name "${KERNEL_NAME}" \
  --display-name "${DISPLAY_NAME}"

echo "Kernel installed successfully."
