"""Logging and Profiling."""

from datetime import datetime
from platform import python_version
from sys import stdout
from time import time as get_time


def _write_log(*msg, end='\n'):
    print(*msg, end=end)


def print_version():
    from . import __version__

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    _write_log(
        f'Running dilimap {__version__} (python {python_version()}) on {date_str}.',
    )


class ProgressReporter:
    def __init__(self, total, interval=3):
        self.count = 0
        self.total = total
        self.timestamp = get_time()
        self.interval = interval

    def update(self):
        self.count += 1
        self.timestamp = get_time()
        percent = int(self.count * 100 / self.total)
        stdout.write(f'\r... {percent}%')
        stdout.flush()

    def finish(self):
        stdout.write('\r')
        stdout.flush()


def profiler(command, filename='profile.stats', n_stats=10):
    """Profiler for a python program (cProfile)

    Stats can be visualized with `!snakeviz profile.stats`.

    Args
        command (str): Command string to be executed.
        filename (str): Name under which to store the stats.
        n_stats (int or None): Number of top stats to show.
    """
    import cProfile
    import pstats

    cProfile.run(command, filename)
    stats = pstats.Stats(filename).strip_dirs().sort_stats('time')
    return stats.print_stats(n_stats or {})
