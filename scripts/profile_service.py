#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import pstats
import runpy


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile a Python entrypoint using cProfile')
    parser.add_argument('module', help='Python module or script path to run under profiler')
    parser.add_argument('--output', default='reports/profile.stats', help='Where to write profile stats')
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    runpy.run_path(args.module, run_name='__main__')
    profiler.disable()
    profiler.dump_stats(args.output)
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
    print(f'profile written to {args.output}')


if __name__ == '__main__':
    main()
