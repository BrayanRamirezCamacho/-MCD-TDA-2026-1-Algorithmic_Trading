sync:
	uv sync

run:
	uv run trading-tda

setup-talib:
	bash scripts/setup_talib.sh

fetch-freqtrade-data: 
	uv run python scripts/fetch_freqtrade_data.py --copy-data
