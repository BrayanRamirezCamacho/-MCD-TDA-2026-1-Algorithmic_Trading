import sys
import json
import shutil
import argparse
import datetime
import subprocess

from loguru import logger

from trading_tda.config import (
    PROJECT_ROOT,
    FREQTRADE_DIR,
)


logger.remove()
logger.add(
    sys.stderr,
    level='DEBUG'
)

TODAY = datetime.date.today()


def default_timerange():
    start = TODAY - datetime.timedelta(days=365)  # 1Year
    end = TODAY

    return f'{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Freqtrade Hyperopt'
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
        '--epochs',
        type=int,
        default=100,
        help='Hyperopt epochs'
    )

    parser.add_argument(
        '--jobs',
        type=int,
        default=2,
        help='Parallel jobs'
    )

    parser.add_argument(
        '--spaces',
        nargs='+',
        default=[
            'buy',
            'sell',
            'stoploss',
            'roi',
            'trailing',
        ],
        help='Hyperopt spaces'
    )

    parser.add_argument(
        '--loss',
        default='SharpeHyperOptLossDaily',
        help='Hyperopt loss function'
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
        'epochs': args.epochs,
        'jobs': args.jobs,
        'spaces': args.spaces,
        'loss': args.loss,
        'command': cmd,
    }

    metadata_path = output_dir / 'hyperopt_metadata.json'

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
        / 'hyperopt'
        / strategy_name
        / args.experiment
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = output_dir / 'run.log'

    cmd = [
        'docker', 'compose',
        '-f', str(FREQTRADE_DIR / 'docker-compose.yml'),
        'run', '--rm',
        'research',
        'hyperopt',
        '--config',
        '/freqtrade/user_data/configs/config_research.json',
        '--strategy',
        args.strategy,
        '--timerange',
        args.timerange,
        '--spaces',
        *args.spaces,
        '-e',
        str(args.epochs),
        '-j',
        str(args.jobs),
        '--hyperopt-loss',
        args.loss,
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

    hyperopt_results_dir = (
        FREQTRADE_DIR
        / 'user_data'
        / 'hyperopt_results'
    )

    for file in hyperopt_results_dir.glob('*'):
        shutil.move(
            str(file),
            output_dir / file.name
        )

    strategy_json = (
        FREQTRADE_DIR
        / 'user_data'
        / 'strategies'
        / f'{strategy_name}.json'
    )

    if strategy_json.exists():
        shutil.copy(
            strategy_json,
            output_dir / strategy_json.name
        )

    export_metadata(
        args=args,
        cmd=cmd,
        output_dir=output_dir
    )

    logger.success('Hyperopt completed')


if __name__ == '__main__':
    main()