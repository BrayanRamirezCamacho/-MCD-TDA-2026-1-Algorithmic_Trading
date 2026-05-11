import sys
import argparse
import datetime
import shutil
import subprocess
import json

from loguru import logger

from trading_tda.config import PROJECT_ROOT, FREQTRADE_DIR


logger.remove()
logger.add(
    sys.stderr,
    level='DEBUG'
)

TODAY = datetime.date.today()


def default_timerange():
    start = TODAY - datetime.timedelta(days=365 * 4)  # default: 4y
    end = TODAY

    return f'{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Freqtrade OHLCV data'
    )

    parser.add_argument(
        '--exchange',
        default='binance',
        help='Exchange name'
    )

    parser.add_argument(
        '--pairs',
        nargs='+',
        default=[
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'SOL/USDT',
            'XRP/USDT',
            'ADA/USDT',
            'DOGE/USDT',
        ],
        help='Trading pairs'
    )

    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['15m', '4h'],
        help='Timeframes'
    )

    parser.add_argument(
        '--timerange',
        default=default_timerange(),
        help='Freqtrade timerange format: YYYYMMDD-YYYYMMDD'
    )

    parser.add_argument(
        '--copy-data',
        action='store_true',
        help='Copy downloaded data into PROJECT_ROOT/data/raw'
    )

    return parser.parse_args()


def copy_data(exchange='binance'):
    src = (
        FREQTRADE_DIR
        / "user_data"
        / "data"
        / exchange
    )

    dst = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / exchange
    )

    if not src.exists():
        raise FileNotFoundError(f'Source data not found: {src}')

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True
    )

    logger.success(f'Copied data: {src} -> {dst}')


def export_metadata(args, cmd, moved_data=False):
    start_date, end_date = args.timerange.split('-')

    metadata_dir = (
        PROJECT_ROOT
        / 'data'
        / 'metadata'
        / f'{args.exchange}_{start_date}_{end_date}'
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    src_data_dir = (
        FREQTRADE_DIR
        / 'user_data'
        / 'data'
        / args.exchange
    )

    dst_data_dir = (
        PROJECT_ROOT
        / 'data'
        / 'raw'
        / args.exchange
    )

    metadata = {
        'created_at': datetime.datetime.now().isoformat(),
        'exchange': args.exchange,
        'pairs': args.pairs,
        'timeframes': args.timeframes,
        'timerange': args.timerange,
        'data_format': 'parquet',
        'moved_data': moved_data,
        'source_data_dir': str(src_data_dir),
        'destination_data_dir': str(dst_data_dir),
        'command': cmd,
    }

    metadata_path = metadata_dir / 'download_metadata.json'

    with open(metadata_path, 'w') as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    logger.success(f'Metadata exported: {metadata_path}')


def main():
    args = parse_args()

    cmd = [
        'docker', 'compose',
        '-f', str(FREQTRADE_DIR / 'docker-compose.yml'),
        'run', '--rm',
        'freqtrade',
        'download-data',
        '--config', 'user_data/config.json',
        '--exchange', args.exchange,
        '--pairs', *args.pairs,
        '--timeframes', *args.timeframes,
        '--timerange', args.timerange,
        '--data-format-ohlcv', 'parquet',
    ]

    logger.info('Starting Freqtrade download:')
    logger.info(f'Exchange: {args.exchange}')
    logger.info(f'Pairs: {args.pairs}')
    logger.info(f'Timeframes: {args.timeframes}')
    logger.info(f'Timerange: {args.timerange}')

    logger.debug(f'\nRunning:\n{" ".join(cmd)}\n')
    subprocess.run(cmd, check=True)

    if args.copy_data:
        copy_data(args.exchange)
    
    export_metadata(
        args=args,
        cmd=cmd,
        moved_data=args.copy_data
    )
    
    logger.success('Download completed')



if __name__ == '__main__':
    main()