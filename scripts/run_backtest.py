import sys
import json
import argparse
import datetime
import subprocess

from loguru import logger

from trading_tda.config import FREQTRADE_DIR


logger.remove()
logger.add(
    sys.stderr,
    level='DEBUG'
)

TODAY = datetime.date.today()


def default_timerange():
    start = TODAY - datetime.timedelta(days=180)
    end = TODAY

    return f'{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Freqtrade Backtesting'
    )

    parser.add_argument(
        '--strategy',
        required=True,
        help='Strategy class name'
    )

    parser.add_argument(
        '--experiment',
        required=True,
        help='Experiment name'
    )

    parser.add_argument(
        '--timerange',
        default=default_timerange(),
        help='Freqtrade timerange'
    )

    parser.add_argument(
        '--timeframe',
        default='15m',
        help='Freqtrade timeframe'
    )

    return parser.parse_args()


def export_metadata(
    args,
    cmd,
    output_dir,
):
    metadata = {
        'created_at': datetime.datetime.now().isoformat(),
        'strategy': args.strategy,
        'experiment': args.experiment,
        'timerange': args.timerange,
        'command': cmd,
    }

    metadata_path = output_dir / 'backtest_metadata.json'

    with open(metadata_path, 'w') as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    logger.success(f'Metadata exported: {metadata_path}')


def run_with_tee(cmd, log_path):
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            log_file.write(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd
            )


def main():
    args = parse_args()

    strategy_name = args.strategy.replace('Strategy', '')

    output_dir = (
        FREQTRADE_DIR
        / 'user_data'
        / 'research'
        / 'backtest'
        / strategy_name
        / args.experiment
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = output_dir / 'report.log'

    cmd = [
        'docker', 'compose',
        '-f', str(FREQTRADE_DIR / 'docker-compose.yml'),
        'run', '--rm',
        'research',
        'backtesting',
        '--config',
        '/freqtrade/user_data/configs/config_research.json',
        '--strategy',
        args.strategy,
        '--timerange',
        args.timerange,
        '--timeframe',
        args.timeframe,
        '--export',
        'trades',
        '--backtest-directory',
        f'/freqtrade/user_data/research/backtest/{strategy_name}/{args.experiment}',
    ]

    logger.info(f'Strategy: {args.strategy}')
    logger.info(f'Experiment: {args.experiment}')
    logger.info(f'Timerange: {args.timerange}')

    logger.debug(f'\nRunning:\n{" ".join(cmd)}\n')

    # subprocess.run(cmd, check=True)
    run_with_tee(
        cmd=cmd,
        log_path=log_path
    )
    export_metadata(
        args=args,
        cmd=cmd,
        output_dir=output_dir
    )

    logger.success('Backtest completed')


if __name__ == '__main__':
    main()