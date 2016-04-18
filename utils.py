# -*- coding: utf-8 -*-
"""
"""
from __future__ import division, print_function
from six import PY2
import os
import sys
import time
from collections import namedtuple
from pprint import pprint
import pandas as pd
from pandas import Timestamp
from config import TIMEZONE

# import xgboost as xgb

def get_time():
    return time.time()


class Timer(object):

    def __init__(self):
        self.timers = {}
        self.first = self.last = get_time()
        self.stack = []

    def stack_str(self):
        return '\nstack=%d\n%s\n' % (
            len(self.stack),
            '\n'.join('%d:%s' % (i, s) for i, s in enumerate(self.stack)))

    def __repr__(self):
        return '\n %d timers\n%s%s' % (
            len(self.timers), '\n'.join(sorted(self.timers.keys())),
            self.stack_str())

    def start(self, key):
        # print('start %d' % len(self.stack), end='')
        assert key not in self.stack, (key, self)
        self.stack.append(key)
        if key not in self.timers:
            self.timers[key] = 0.0, 0, None, None
        d, n, s, e = self.timers[key]
        if not (s is None and e is None):
            print('!!!!!!', (key, (d, n, s, e), self))
            print('$$$$$$$$$$$')
        assert s is None and e is None, ('@@', key, (d, n, s, e), self)
        self.timers[key] = d, n, get_time(), e
        # print('--->', key, len(self.stack))

    def end(self, key):
        # print('end  ---<', key, len(self.stack), end='')
        assert key in self.timers, (key, self)
        assert key == self.stack[-1], ([key, self.stack[-1]], self)
        d, n, s, e = self.timers[key]
        assert s is not None, (key, self.timers[key], self)
        assert e is None, (key, self.timers[key], self)
        e = get_time()
        assert e >= s, (s, e)
        n += 1
        d += e - s
        self.timers[key] = d, n, None, None
        self.last = e
        # print('%d timers. %.1f sec' % (len(self.timers), self.last - self.first))
        self.stack.pop()
        # print('!!!! %d' % len(self.stack))

    def show(self):
        duration = self.last - self.first
        print('`' * 80)
        print('%d timers. %.1f sec' % (len(self.timers), duration))
        for key in sorted(self.timers, key=lambda k: self.timers[k])[::-1]:
            d, n, s, e = self.timers[key]
            f = (d / duration) if duration else 0.0
            print('%s: %4.1f (%2d%%) %d' % (key, d, int(round(f * 100)), n))
        print('N_BLAME_PROCESSES=%d, DO_MULTIPROCESSING=%s' % (N_BLAME_PROCESSES,
              DO_MULTIPROCESSING))

_timer = Timer()


def timer(func):
    """
        Outputs the time a function takes
        to execute.
    """
    from functools import wraps
    name = 'func::%s' % (func.__name__ if PY2 else func.__qualname__)
    # print(func.__name__, func.__qualname__)

    @wraps(func)
    def decorated_function(*args, **kwargs):
        _timer.start(name)
        exception = None

        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, (TypeError, AssertionError)):
                raise
            print('&&& timer: %s: caught %s:%s %s' % (name, type(e), e,
                _timer.stack_str()))
            exception = e
            if isinstance(e, AssertionError):
                raise
            while _timer.stack and _timer.stack[-1] != name:
                print('&&! Unwinding %d:%s%s' % (
                      len(_timer.stack),
                      _timer.stack[-1],
                      _timer.stack_str()))
                _timer.end(_timer.stack[-1])
                # _timer.stack.pop()
            if not _timer.stack:
                print('Noooo timer stack', _timer.stack_str())
            # sptr = _timer.stack.index(name)
            # assert sptr >= 0
            # print('&&& stack: %d: %s->%s' % (sptr, _timer.stack, _timer.stack[:sptr + 1]))
            # _timer.stack, _timer.stack[:sptr + 1]

        _timer.end(name)
        if exception is not None:
            raise exception
        return ret
    return decorated_function


class ProcessPool(object):
    """Package of Pool and ThreadPool for 'with' usage.
    """
    SINGLE = 0
    THREAD = 1
    PROCESS = 2

    def __init__(self, process_type, n_pool):
        if not DO_MULTIPROCESSING:
            process_type = ProcessPool.SINGLE
        self.process_type = process_type
        if process_type != ProcessPool.SINGLE:
            clazz = ThreadPool if process_type == ProcessPool.THREAD else Pool
            self.pool = clazz(n_pool)

    def __enter__(self):
        return self

    def imap_unordered(self, func, args_iter):
        if self.process_type != ProcessPool.SINGLE:
            return self.pool.imap_unordered(func, args_iter)
        else:
            return map(func, args_iter)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process_type != ProcessPool.SINGLE:
            self.pool.terminate()


def date_str(date):
    """The way we show dates in reports."""
    return date.strftime('%Y-%m-%d')

DAY = pd.Timedelta('1 days')  # 24 * 3600 * 1e9 in pandas nanosec time

# Max date accepted for commits. Clearly a sanity check
MAX_DATE = Timestamp('today').tz_localize(TIMEZONE) + DAY


def to_timestamp(date_s):
    """Convert string `date_s' to pandas Timestamp in `TIMEZONE`
        NOTE: The idea is to get all times in one timezone.
    """
    return Timestamp(date_s).tz_convert(TIMEZONE)


def delta_days(t0, t1):
    """Returns: time from `t0` to `t1' in days where t0 and t1 are Timestamps
        Returned value is signed (+ve if t1 later than t0) and fractional
    """
    return (t1 - t0).total_seconds() / 3600 / 24

concat = ''.join
path_join = os.path.join

Commit = namedtuple('Commit', ['sha', 'author', 'date', 'merge1', 'merge2', 'body', 'issue'])
# print(Commit._fields)
# assert False


def show_commit(commit, title=None):
    assert len(Commit._fields) == len(commit), (len(Commit._fields), len(commit))
    lines = ['%6s: %s' % (key, val) for key, val in zip(Commit._fields, commit) if val is not None]
    if title:
        title += (80 - len(title)) * '*'
        lines = [title] + lines
    return '\n'.join(lines)


@timer
def save_object(path, obj):
    """Save object `obj` to `path` after bzipping it
    """
    # existing_pkl is for recovering from bad pickles
    existing_pkl = '%s.old.pkl' % path
    if os.path.exists(path) and not os.path.exists(existing_pkl):
        os.rename(path, existing_pkl)

    with bz2.BZ2File(path, 'w') as f:
        # protocol=2 makes pickle usable by python 2.x
        pickle.dump(obj, f, protocol=2)

    # Delete existing_pkl if load_object succeeds
    load_object(path)
    if os.path.exists(path) and os.path.exists(existing_pkl):
        os.remove(existing_pkl)


@timer
def load_object(path, default=None):
    """Load object from `path`
    """
    if default is not None and not os.path.exists(path):
        return default
    try:
        with bz2.BZ2File(path, 'r') as f:
            return pickle.load(f)
    except:
        print('load_object(%s, %s) failed' % (path, default), file=sys.stderr)
        raise


def save(path, obj):
    print('Saving to "%s"' % path)
    with open(path, 'wt') as f:
        pprint(obj, stream=f)


def load(path, default):
    if not os.path.exists(path):
        return default
    with open(path, 'rt') as f:
        text = f.read()
    return eval(text)


def mkdir(path):
    """Create directory `path` including all intermediate-level directories and ignore
        "already exists" errors.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise


def normalize_path(path):
    """Returns: `path` without leading ./ and trailing / . \ is replaced by /
    """
    path = RE_SLASH.sub('/', path)
    if path.startswith('./'):
        path = path[2:]
    if path.endswith('/'):
        path = path[:-1]
    return path


def clean_path(path):
    """Returns: `path` with characters that are illegal in filenames replaced with '_'
    """
    return RE_PATH.sub('_', normalize_path(path))

