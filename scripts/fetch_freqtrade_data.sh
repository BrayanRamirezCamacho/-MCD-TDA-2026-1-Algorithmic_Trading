#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FT_DIR="$ROOT_DIR/freqtrade"

PS="BTC/USDT ETH/USDT"
TFS="15m 4h"
TRS="20251001-20260510"


EXCHANGE=${1:-binance}
PAIRS=${2:-$PS}
TIMEFRAMES=${3:-$TFS}
TIMERANGE=${4:-$TRS}

docker compose -f "$FT_DIR/docker-compose.yml" run --rm freqtrade download-data \
  --config user_data/config.json \
  --exchange "$EXCHANGE" \
  --pairs $PAIRS \
  --timeframes $TIMEFRAMES \
  --timerange $TIMERANGE \
  --data-format-ohlcv parquet