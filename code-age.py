# -*- coding: utf-8 -*-
"""
    Analyze code age in a git repository

    Writes reports in the following locations

    e.g. For repository "Linux"

    [root]
        │
        ├── linux
        │   └── reports
        │       └── 2015-11-25.6ffeba96
        │           ├── __sh
        │           │   ├── Adrian_Hunter
        │           │   │   ├── history[all].png
        │           │   │   ├── history[all].txt
        │           │   │   └── oldest[all].txt
        │           │   ├── Akinobu_Mita
        │           │   │   ├── history[all].png
        │           │   │   ├── history[all].txt
        │           │   │   └── oldest[all].txt

    _[root]_ defaults to ~/git.stats
    _"linux"_ is the remote name of the repository being analyzed. It was extracted from
            https://github.com/torvalds/linux.git
    _"2015-11-25.6ffeba96"_ contains reports on revision 6ffeba96. 6ffeba96 was
            created on 2015-11-25. We hope that putting the date in the directory name makes it
            easier to navigate.
    _"__sh"_ is the directory containing reports on *.sh files
    _Adrian_Hunter_ is the directory containing reports on author Adrian_Hunter files in revision 6ffeba96

"""
from __future__ import division, print_function
import subprocess
from collections import defaultdict, Counter
import sys
import time
import re
import os
import stat
import glob
import errno
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Timestamp
import matplotlib
import matplotlib.pylab as plt
from matplotlib.pylab import cycler
import bz2

# Python 2 / 3 stuff
try:
    import cPickle as pickle
except:
    import pickle
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


#
# Configuration.
#
TIMEZONE = 'Australia/Melbourne'
HASH_LEN = 8
STRICT_CHECKING = False


# Set graphing style
matplotlib.style.use('ggplot')
plt.rcParams['axes.prop_cycle'] = cycler('color', ['b', 'y', 'k', '#707040', '#404070'])
plt.rcParams['savefig.dpi'] = 300

try:
    PATH_MAX = os.pathconf(__file__, 'PC_NAME_MAX')
except:
    PATH_MAX = 255

IGNORED_EXTS = {
    '.air', '.bin', '.bmp', '.cer', '.cert', '.der', '.developerprofile', '.dll', '.doc', '.docx',
    '.exe', '.gif', '.icns', '.ico', '.jar', '.jpeg', '.jpg', '.keychain', '.launch', '.pdf',
    '.pem', '.pfx', '.png', '.prn', '.so', '.spc', '.svg', '.swf', '.tif', '.tiff', '.xls', '.xlsx'
   }


def is_windows():
    try:
        sys.getwindowsversion()
    except:
        return False
    else:
        return True


def lowpriority():
    """ Set the priority of the process to below-normal.
        http://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
    """
    if is_windows():
        import win32api
        import win32process
        import win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)

    else:
        os.nice(1)


def truncate_hash(hsh):
    """The way we show hashes"""
    return hsh[:HASH_LEN]


def date_str(date):
    """The way we show dates"""
    return date.strftime('%Y-%m-%d')

DAY = pd.Timedelta('1 days')  # 24 * 3600 * 1e9

# Max date accepted for commits. Clearly a sanity check
MAX_DATE = Timestamp('today').tz_localize(TIMEZONE) + DAY


def to_timestamp(date_s):
    """Convert string `date_s' to pandas Timestamp in `TIMEZONE`
        NOTE: The idea is to get all timess in one timezone.
    """
    return Timestamp(date_s).tz_convert(TIMEZONE)


def delta_days(t0, t1):
    """Return time from `t0` to `t1' in days where t0 and t1 are Timestamps
        Returned value is signed (+ve if t1 later than t0) and fractional
    """
    return (t1 - t0).total_seconds() / 3600 / 24

concat = ''.join


def save_object(path, obj):
    """Save `obj` to `path`"""
    # old is for recovering from bad pickles
    existing_pkl = '%s.old.pkl' % path
    if os.path.exists(path) and not os.path.exists(existing_pkl):
        os.rename(path, existing_pkl)

    with bz2.BZ2File(path, 'w') as f:
        pickle.dump(obj, f)

    # Delete old if load_object succeeds
    load_object(path)
    if os.path.exists(path) and os.path.exists(existing_pkl):
        os.remove(existing_pkl)


def load_object(path, default=None):
    """Load object from `path`"""
    if default is not None and not os.path.exists(path):
        return default
    try:
        try:
            with bz2.BZ2File(path, 'r') as f:
                return pickle.load(f)
        except:
            with open(path, 'rb') as f:
                return pickle.load(f)
    except:
        print('load_object(%s, %s) failed' % (path, default), file=sys.stderr)
        raise


def mkdir(path):
    """Create directory `path` and ignore "already exists" errors
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise
    assert os.path.exists(path), path  # !@#$ now superfluous


def df_append_totals(df_in):
    """Append row and column totals to Pandas DataFrame `df_in`, remove all zero columns and sort
        rows and columns by total.
    """
    assert 'Total' not in df_in.index
    assert 'Total' not in df_in.columns
    rows, columns = list(df_in.index), list(df_in.columns)
    df = DataFrame(index=rows + ['Total'], columns=columns + ['Total'])
    df.iloc[:-1, :-1] = df_in
    df.iloc[:, -1] = df.iloc[:-1, :-1].sum(axis=1)
    df.iloc[-1, :] = df.iloc[:-1, :].sum(axis=0)
    row_order = ['Total'] + sorted(rows, key=lambda r: -df.loc[r, 'Total'])
    column_order = ['Total'] + sorted(columns, key=lambda c: -df.loc['Total', c])
    df = df.reindex_axis(row_order, axis=0)
    df = df.reindex_axis(column_order, axis=1)
    empties = [col for col in df.columns if df.loc['Total', col] == 0]
    df.drop(empties, axis=1, inplace=True)
    return df


def moving_average(series, window):
    """Return weighted moving average of Pandas Series `series`
        Weights are a triangle of width `window`
    """
    n = len(series)
    d0 = (window) // 2
    d1 = window - d0
    weights = np.empty(window, dtype=np.float)
    radius = (window - 1) / 2
    for i in range(window):
        weights[i] = radius + 1 - abs(i - radius)

    n = len(series)
    ma = np.empty(n, dtype=float)
    for i in range(n):
        i0 = max(0, i - d0)
        i1 = min(n, i + d1)
        c0 = i0 - (i - d0)
        c1 = (i + d1) - n

        v = np.average(series[i0:i1], weights=weights[c0:window - c1])
        ma[i] = min(max(series[i0:i1].min(), v), series[i0:i1].max())

    return Series(ma, index=series.index)


def print_fit(s, width=100):
    """Print string `s` in `width` chars, removing middle characters if necessary"""
    width = max(20, width)
    if len(s) > width:
        notch = int(round(width * 0.75)) - 5
        end = width - 5 - notch
        return '%s ... %s' % (s[:notch], s[-end:])
    return s


def get_ext(path):
    parts = os.path.splitext(path)
    return parts[-1] if parts else '[None]'


def exec_output(command, require_output):
    # TODO save stderr and print it on error
    try:
        output = subprocess.check_output(command)
    except:
        print('exec_output failed: command=%s' % command, file=sys.stderr)
        raise
    if require_output and not output:
        raise RuntimeError('exec_output: command=%s' % command)
    return decode_str(output)


def exec_output_lines(command, require_output):
    return exec_output(command, require_output).splitlines()


def exec_headline(command):
    return exec_output(command, True).splitlines()[0]


def git_file_list(path_patterns=()):
    return exec_output_lines(['git', 'ls-files'] + path_patterns, False)


def git_diff(rev1, rev2):
    return exec_output_lines(['git', 'diff', '--name-only', rev1, rev2], False)


def git_show_oneline(obj):
    """https://git-scm.com/docs/git-show
    """
    return exec_headline(['git', 'show', '--oneline', '--quiet', obj])


def git_date(obj):
    date_s = exec_headline(['git', 'show', '--pretty=format:%ai', '--quiet', obj])
    return to_timestamp(date_s)


RE_REMOTE_URL = re.compile(r'(https?://.*/[^/]+(?:\.git)?)\s+\(fetch\)')

RE_REMOTE_NAME = re.compile(r'https?://.*/(.+?)(\.git)?$')


def git_remote():
    """Returns: remote_url, remote_name
    """
    # $ git remote -v
    # origin  https://github.com/FFTW/fftw3.git (fetch)
    # origin  https://github.com/FFTW/fftw3.git (push)

    for line in exec_output_lines(['git', 'remote', '-v'], True):
        m = RE_REMOTE_URL.search(line)
        if not m:
            continue
        remote_url = m.group(1)
        remote_name = RE_REMOTE_NAME.search(remote_url).group(1)
        return remote_url, remote_name

    raise RuntimeError('No remote')


def git_describe():
    return exec_headline(['git', 'describe', '--always'])


def git_name():
    """Returns name of current revision.
    """
    return ' '.join(exec_headline(['git', 'name-rev', 'HEAD']).split()[1:])


def git_current_branch():
    branch = exec_headline(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == 'HEAD':  # Detached HEAD?
        branch = None
    return branch


def git_current_revision():
    return exec_headline(['git', 'rev-parse', 'HEAD'])


RE_PATH = re.compile('''[^a-z^0-9^!@#$\-+=_\[\]\{\}\(\)]''', re.IGNORECASE)
RE_SLASH = re.compile(r'[\\/]+')


def normalize_path(path):
    path = RE_SLASH.sub('/', path)
    if path.startswith('./'):
        path = path[2:]
    if path.endswith('/'):
        path = path[:-1]
    return path


def clean_path(path):
    return RE_PATH.sub('_', normalize_path(path))


def git_blame_text(path):
    """Return git blame text for file `path`
    """
    return exec_output(['git', 'blame', '-l', '-f', '-w', '-M', path], False)


RE_BLAME = re.compile(b'''
                      \^*([0-9a-f]{4,})\s+
                      .+?\s+
                      \(
                      (.+?)\s+
                      (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})
                      \s+(\d+)
                      \)''',
                      re.DOTALL | re.MULTILINE | re.VERBOSE)


class GitException(Exception):
    pass


if is_windows():
    RE_LINE = re.compile(b'(?:\r\n)+')
else:
    RE_LINE = re.compile(b'[\n]+')


def _update_text_hash_loc(hash_date_author, path_hash_loc, path_set, author_set, ext_set,
    text, path, max_date=None):
    """Update key data structures with information that we parse from `text` in this function
    """

    if max_date is None:
        max_date = MAX_DATE

    assert max_date <= MAX_DATE, max_date

    hash_loc = Counter()

    lines = RE_LINE.split(text)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        print('    %s is empty' % path, file=sys.stderr)
        raise GitException

    for i, ln in enumerate(lines):
        if not ln:
            continue

        m = RE_BLAME.match(ln)
        if not m:
            assert False, ln
            raise GitException

        if m.group(2) == 'Not Committed Yet':
            continue
        hsh = m.group(1)
        author = m.group(2)
        date_s = m.group(3)
        line_n = int(m.group(4))

        author = author.strip()
        if author == '':
            author = '<>'

        assert line_n == i + 1, 'line_n=%d,i=%d\n%s\n%s' % (line_n, i, path, m.group(0))
        assert author.strip() == author, 'author="%s\n%s:%d\n%s",' % (
                                 author, path, i + 1, ln[:200])

        date = to_timestamp(date_s)

        if date > max_date:
            print('Bogus timestamp: %s:%d\n%s' % (path, i + 1, ln[:200]), file=sys.stderr)
            continue

        hash_loc[hsh] += 1
        hash_date_author[hsh] = (date, author)
        author_set.add(author)

    assert all(hsh in hash_date_author for hsh in hash_loc), path

    path_hash_loc[path] = hash_loc
    if STRICT_CHECKING:
        assert path in path_hash_loc, path
        assert len(path_hash_loc[path]) > 0, path
        assert sum(path_hash_loc[path].values()) > 0, path


def get_ignored_files(gitstatsignore):
    if gitstatsignore is None:
        gitstatsignore = 'gitstatsignore'
    else:
        assert os.path.exists(gitstatsignore), 'gitstatsignore file "%s"' % gitstatsignore

    if not gitstatsignore or not os.path.exists(gitstatsignore):
        return set()

    ignored_files = set()
    with open(gitstatsignore, 'rt') as f:
        for line in f:
            line = line.strip('\n').strip()
            if not line:
                continue
            ignored_files.update(git_file_list([line]))

    return ignored_files


def ts_summary(ts, desc=None):
    """Return a summary of time series `ts`
    """
    if len(ts) == 1:
        return '[singleton]'

    if STRICT_CHECKING:
        assert len(ts) > 1, (ts.index[0], ts.index[-1], desc)
        assert ts.index[0] < ts.index[-1], (ts.index[0], ts.index[-1])
        assert ts.index[0] <= ts.argmax(), (ts.index[0], ts.argmax())
        assert ts.argmax() <= ts.index[-1], (ts.argmax(), ts.index[-1])

    return ('min=%.1f,max=%.1f=%.2f,median=%.1f,mean=%.1f=%.2f,'
            '[tmin,argmax,tmax]=%s' % (
             ts.min(), ts.max(), ts.max() / ts.sum(),
             ts.median(), ts.mean(), ts.mean() / ts.sum(),
             ','.join([date_str(t) for t in (ts.index[0], ts.argmax(), ts.index[-1])])
            ))


def find_peaks(ts):
    """Find peaks in time series `ts`
    """
    from scipy import signal

    MIN_PEAK_DAYS = 60   # !@#$ Reduce this
    MAX_PEAK_DAYS = 1

    # !@#$ Tune np.arange(2, 10) * 10 to data
    peak_idx = signal.find_peaks_cwt(ts, np.arange(2, 10) * 10)

    return [i for i in peak_idx
            if (delta_days(ts.index[0], ts.index[i]) >= MIN_PEAK_DAYS and
                delta_days(ts.index[i], ts.index[-1]) >= MAX_PEAK_DAYS)
            ]


#
# Time series analysis
#  !@#$ try different windows to get better peaks
def analyze_time_series(loc, date, window=60):
    """Return a histogram of LoC / day for events given by `loc` and `date`
        loc: list of LoC events
        date: list of timestamps for loc
        n_peaks: max number of peaks to find
        window: width of weighted moving average window used to smooth data
        Returns: averaged time series, list of peaks in time series
    """
    # ts is histogram of LoC with 1 day bins. bins start at midnight on TIMEZONE
    # !@#$ maybe offset dates in ts to center of bin (midday)

    ts = Series(loc, index=date)  # !@#$ dates may not be unique, guarantee this
    ts_days = ts.resample('D', how='mean')  # Day
    ts_days = ts_days.fillna(0)

    # tsm is smoothed ts
    ts_ma = moving_average(ts_days, window) if window else ts_days

    peak_idx = find_peaks(ts_ma)

    return ts_ma, peak_idx


def make_history(author_date_loc, n_peaks, author_list=None):
    """Return a history for all authors in `author_list`
    """
    # date_loc could be a Series !@#$
    # for author, date_loc in author_date_loc.items():
    #     date, loc = zip(*date_loc)
    #     assert delta_days(min(date), max(date)) > 30, (author, min(date), max(date))

    loc, date = [], []
    if author_list is None:
        author_list = author_date_loc.keys()
    for author in author_list:
        assert author_date_loc[author]
        d, l = zip(*author_date_loc[author])
        date.extend(d)
        loc.extend(l)
    # assert delta_days(min(date), max(date)) > 30, (author, min(date), max(date))
    ts, peak_idx = analyze_time_series(loc, date)

    peak_pxy = [(p, ts.index[p], ts.iloc[p]) for p in peak_idx]

    def key_pxy(p, x, y):
        return -y, x, p

    peak_pxy.sort(key=lambda k: key_pxy(*k))
    peak_ixy = [(i, x, y) for i, (p, x, y) in enumerate(peak_pxy[:n_peaks])]

    return ts, tuple(peak_ixy)


def key_ixy_i(i, x, y):
    return i


def key_ixy_x(i, x, y):
    return x, i


def _get_xy_text(xy_plot, txt_width, txt_height):
    """Return  positions of text labels for points `xy_plot`
        1) Offset text upwards by txt_width
        2) Remove collisions
        NOTE: xy_plot MUST be sorted by y when this function is called

        !@#$ 2 pass solution
        1: Move each point above highest lower colllider. track original y
        2: go through all raised points (lowest to highest), move point to lowest gap, if any
            betweeen y_orig..y_new

        Based on http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
    """

    # More natural with ndarray's !@#$
    yx_plot = [[y + txt_height, x] for x, y in xy_plot]
    yx_plot0 = [[y + txt_height, x] for x, y in xy_plot]

    # Working from bottom to top
    for i, (y, x) in enumerate(yx_plot):
        # yx_collisions is all points that collide with yx_plot[i], lowest first
        yx_collisions = sorted((yy, xx) for yy, xx in yx_plot
                               if yy > y - txt_height and
                               abs(xx - x) < txt_width * 2 and
                               (yy, xx) != (y, x))
        if not yx_collisions or abs(yx_collisions[0][0] - y) >= txt_height:
            continue

        # yx_plot[i] is colliding with yx_collisions[0], the lowest colliding point

        # Search for lowest space between collisions that is big enough for text
        for j, (dy, dx) in enumerate(np.diff(yx_collisions, axis=0)):
            if dy > txt_height * 2:  # Room for text?
                yx_plot[i][0] = yx_collisions[j][0] + txt_height
                break
        else:
            # move yx_plot[i] above yx_collisions[-1], the highest colliding point
            yx_plot[i][0] = yx_collisions[-1][0] + txt_height

    # Move points down to any gaps that have opened in above moves

    # try 5 times to give points a chance to settle
    for n in range(100):
        n_changes = 0
        changed = False
        yx_plot1 = [(y, x) for y, x in yx_plot]

        for i, ((y0, x0), (y, x)) in enumerate(zip(yx_plot0, yx_plot)):
            if y == y0:
                continue
            yx_collisions = sorted((yy, xx) for yy, xx in yx_plot
                                   if y0 - txt_height <= yy < y + txt_height and
                                   abs(xx - x) < txt_width * 2 and
                                   (yy, xx) != (y, x))
            if not yx_collisions:
                yx_plot[i][0] = y0
                changed = True
                n_changes += 1

            elif yx_collisions[0][0] > y0 + txt_height * 2:
                yx_plot[i][0] = y0
                changed = True
                n_changes += 1

            else:
                yx_collisions2 = [(y0, x0)] + yx_collisions + [(y, x)]
                for j, (dy, dx) in enumerate(np.diff(yx_collisions2, axis=0)):
                    if dy > txt_height * 2:  # Room for text?
                        yx_plot[i][0] = yx_collisions2[j][0] + txt_height
                        changed = True
                        n_changes += 1

                        break
            if not changed and y > yx_collisions[-1][0] + 2 * txt_height:
                yx_plot[i][0] = yx_collisions[-1][0] + 2 * txt_height
                changed = True
                n_changes += 1

        # print('**** n=%d,n_changes=%d,changed=%s' % (n, n_changes, changed))
        for i, ((y0, x0), (y, x), (y1, x1)) in enumerate(zip(yx_plot0, yx_plot, yx_plot1)):
            assert y0 <= y <= y1, (y0, y, y1)
        if not changed:
            break

    return [(x, y) for y, x in yx_plot]


def plot_loc_date(ax, label, history):
    """Plot LoC vs date for time series in history
    """
    # TODO Show areas !@#$
    tsm, peak_ixy = history

    tsm.plot(label=label, ax=ax)

    X0, X1, Y0, Y1 = ax.axis()

    if not peak_ixy:
        return
    peak_ixy = sorted(peak_ixy)

    x0 = tsm.index[0]
    # !@#$ TODO Get actual text size
    txt_height = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.15 * (plt.xlim()[1] - plt.xlim()[0])

    # NOTE: The following code assumes xy_data is sorted by y

    xy_plot = [(delta_days(x0, x) + X0, y) for _, x, y in peak_ixy]
    xy_text = _get_xy_text(xy_plot, txt_width, txt_height)
    plt.ylim(plt.ylim()[0], max(plt.ylim()[1], max(y for _, y in xy_text) + 2 * txt_height))
    X0, X1, Y0, Y1 = ax.axis()

    # Write high labels first so that lower label text overwrites higher label arrows
    def key_text(ixy, xy_p, xy_t):
        (i, x, y), (x_p, y_p), (x_t, y_t) = ixy, xy_p, xy_t
        return -y_t, x_t, i

    i_data_plot_text = sorted(zip(peak_ixy, xy_plot, xy_text), key=lambda k: key_text(*k))

    # Label the peaks
    for (i, x, y), (x_p, y_p), (x_t, y_t) in i_data_plot_text:
        ax.annotate('%d) %s, %.0f' % (i + 1, date_str(x), y),
                    xy=(x_p, y_p), xytext=(x_t, y_t),
                    arrowprops={'facecolor': 'red'},
                    horizontalalignment='center', verticalalignment='bottom')
        assert X0 <= x_p <= X1, (X0, x_p, X1)
        assert X0 <= x_t <= X1, (X0, x_t, X1)
        assert Y0 <= y_t <= Y1, (Y0, y_t, Y1)


def plot_show(ax, blame_map, report_map, author, do_show, graph_path, do_legend):
    """Show and/or save the current markings in axes `ax`
    """

    repo_summary = blame_map._repo_map.summary
    rev_summary = blame_map._rev_map.summary
    path_patterns = report_map.path_patterns

    path_str = ''
    if len(path_patterns) == 1:
        path_str = '/%s' % path_patterns[0]
    elif len(path_patterns) > 1:
        path_str = '/[%s]' % '|'.join(path_patterns)

    if do_legend:
        plt.legend(loc='best')

    ax.set_title('%s%s code age (as of %s)\n'
                 'revision=%s : "%s", author: %s' % (
                  repo_summary['remote_name'],
                  path_str,
                  date_str(rev_summary['date']),
                  truncate_hash(rev_summary['revision_hash']),
                  blame_map.get_description(),
                  author
                 ))
    ax.set_xlabel('date')
    ax.set_ylabel('LoC / day')
    if graph_path:
        plt.savefig(graph_path)
    if do_show:
        plt.show()
        assert False


def make_data_dir(path):
    return os.path.join(path, 'data')


class Persistable(object):
    """Saves a catalog of objects and a summary and manifest to disk
    """

    def make_path(self, name):
        return os.path.join(self.base_dir, name)

    def __init__(self, summary, base_dir, **kwargs):
        assert 'TEMPLATE' in self.__class__.__dict__, self.__class__.__dict__
        self.base_dir = base_dir
        self.data_dir = make_data_dir(base_dir)
        self.summary = summary.copy()
        self.catalog = {k: kwargs.get(k, v()) for k, v in self.__class__.TEMPLATE.items()}

    def load(self):
        catalog = load_object(self.make_path('data.pkl'), {})
        catalog = {k: v for k, v in catalog.items() if k in self.catalog}
        for k, v in catalog.items():
            self.catalog[k].update(v)
        path = os.path.join(self.base_dir, 'summary')
        if os.path.exists(path):
            self.summary = eval(open(path, 'rt').read())

    def save(self, force):
        # Load before saving in case another instance of this script is running
        path = self.make_path('data.pkl')
        if not force and os.path.exists(path):
            catalog = load_object(path, {})
            catalog = {k: v for k, v in catalog.items() if k in self.catalog}
            for k, v in self.catalog.items():
                # assert k in catalog, (k, catalog.keys(), path)
                if k in catalog:
                    catalog[k].update(v)
                else:
                    catalog[k] = v
            self.catalog = catalog

        mkdir(self.base_dir)

        save_object(path, self.catalog)
        print('saved %s' % path)

        path = self.make_path('summary')
        open(os.path.join(path), 'wt').write(repr(self.summary))
        assert os.path.exists(path), os.path.abspath(path)

        manifest = {k: len(v) for k, v in self.catalog.items()}
        path = self.make_path('manifest')
        open(os.path.join(path), 'wt').write(repr(manifest))
        assert os.path.exists(path), os.path.abspath(path)

    def __repr__(self):
        return repr([self.base_dir, {k: len(v) for k, v in self.catalog.items()}])


class BlameRepoMap(Persistable):
    """Repository level persisted data structures
        Currently this is hash_date_author.
    """
    TEMPLATE = {'hash_date_author': lambda: {}}


class BlameRevMap(Persistable):
    """Revision level persisted data structures
        The main structure is path_hash_loc.
    """

    TEMPLATE = {
        'path_hash_loc': lambda: {},
        'path_set': lambda: set(),
        'bad_files': lambda: set(),
        'author_set': lambda: set(),
        'ext_set': lambda: set(),
    }


class BlameMap(object):
    """A BlameMap contains data from git blame that are used to compute reports
        This data can take a long time to generate so we allow it to be saved to and loaded from
        disk so that it can be reused between runs
    """

    def __init__(self, repo_base_dir, repo_summary, rev_summary):
        self._repo_base_dir = repo_base_dir
        self.repo_dir = os.path.join(repo_base_dir, 'cache')
        self._repo_map = BlameRepoMap(repo_summary, self.repo_dir)
        # self.rev_dir = os.path.join(self._repo_map.base_dir, truncate_hash(rev_summary['commit']))
        rev_dir = os.path.join(self._repo_map.base_dir, truncate_hash(rev_summary['revision_hash']))
        self._rev_map = BlameRevMap(rev_summary, rev_dir)

    def copy(self, rev_dir):
        """Return a copy of self with its rev_dit replaced
        """
        blame_map = BlameMap(self._repo_base_dir, self._repo_map.summary, self._rev_map.summary)
        # blame_map.rev_dir = rev_dir              # !@#$ which of these neeeds to be set to rev_dir
        blame_map._rev_map.base_dir = rev_dir
        return blame_map

    def load(self):
        self._repo_map.load()
        self._rev_map.load()
        return self

    def save(self, force):
        self._repo_map.save(force)
        self._rev_map.save(force)

    def __repr__(self):
        return repr({k: repr(v) for k, v in self.__dict__.items()})

    @property
    def hash_date_author(self):
        return self._repo_map.catalog['hash_date_author']

    @property
    def path_hash_loc(self):
        return self._rev_map.catalog['path_hash_loc']

    @property
    def path_set(self):
        return self._rev_map.catalog['path_set']

    @property
    def bad_files(self):
        return self._rev_map.catalog['bad_files']

    def get_description(self):
        """Our best guess at describing the current revision"""
        summary = self._rev_map.summary
        description = summary.get('branch')
        if not description:
            description = summary['description']
        return description

    def _get_peer_revisions(self):
        peer_dirs = [rev_dir for rev_dir in glob.glob(os.path.join(self.repo_dir, '*'))
                     if rev_dir != self._rev_map.base_dir and
                     os.path.exists(os.path.join(rev_dir, 'data.pkl'))]

        for rev_dir in peer_dirs:
            rev = self.copy(rev_dir).load()
            yield rev_dir, rev

    def _update_from_existing(self, file_list):

        remaining_path_set = set(file_list) - self.path_set

        print('-' * 80)
        print('Update data from previous blames. %d remaining of %d files' % (
              len(remaining_path_set), len(file_list)))

        if not remaining_path_set:
            return

        peer_revisions = list(self._get_peer_revisions())
        print('Checking up to %d peer revisions for blame data' % len(peer_revisions))

        this_hash = self._rev_map.summary['revision_hash']  # !@#$% commit => revision

        for i, (rev_dir, that_rev) in enumerate(peer_revisions):

            if not remaining_path_set:
                break

            print('%2d: %s,' % (i, rev_dir), end=' ')

            that_hash = that_rev._rev_map.summary['revision_hash']
            that_path_set = that_rev.path_set
            that_bad_files = that_rev.bad_files
            diff_set = set(git_diff(this_hash, that_hash))

            self.bad_files.update(that_bad_files - diff_set)
            existing_path_set = remaining_path_set & (that_path_set - diff_set)

            for path in existing_path_set:
                assert path in that_path_set
                if path in that_rev.path_hash_loc:
                    self.path_hash_loc[path] = that_rev.path_hash_loc[path]

                self.path_set.add(path)
                remaining_path_set.remove(path)
            print('%d files of %d remaining, diff=%d' % (
                   len(remaining_path_set), len(file_list), len(diff_set)))

    def _update_new_files(self, file_list, force):
        """Compute base statistics over whole revision
            blame all files in `path_patterns`
            Update: hash_date_author, path_hash_loc for files that are not already in path_hash_loc

            Also update exts_good, exts_good_loc, exts_bad, exts_ignored
        """

        rev_summary = self._rev_map.summary
        hash_date_author = self.hash_date_author
        path_set = self.path_set
        author_set = self._rev_map.catalog['author_set']
        ext_set = self._rev_map.catalog['ext_set']

        path_hash_loc = self._rev_map.catalog['path_hash_loc']

        assert isinstance(hash_date_author, dict), type(hash_date_author)

        if not force:
            file_list = [path for path in file_list if path not in path_set]
        n_files = len(file_list)
        print('-' * 80)
        print('Update data by blaming %d files' % len(file_list))

        paths0 = len(path_set)
        loc0 = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
        commits0 = len(hash_date_author)
        start = time.time()
        blamed = 0
        last_loc = loc0
        last_i = 0

        for i, path in enumerate(file_list):

            path_set.add(path)

            if os.path.basename(path) in {'.gitignore'}:
                self.bad_files.add(path)
                continue

            if i - last_i >= 100:
                duration = time.time() - start

                loc = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
                if loc != last_loc:
                    rate = blamed / duration if duration >= 1.0 else 0
                    print('i=%d of %d(%.1f%%),files=%d,loc=%d,commits=%d,dt=%.1f,r=%.1f,path=%s' % (
                          i, n_files, 100 * i / n_files, blamed,
                          loc - loc0,
                          len(hash_date_author) - commits0,
                          duration, rate,
                          print_fit(path)))
                    sys.stdout.flush()
                    last_loc = loc
                    last_i = i

            try:
                max_date = rev_summary['date']
                text = git_blame_text(path)
                _update_text_hash_loc(hash_date_author, path_hash_loc, path_set, author_set,
                                      ext_set, text, path, max_date)
            except GitException:
                apath = os.path.abspath(path)
                self.bad_files.add(path)
                if not os.path.exists(path):
                    print('    %s no longer exists' % apath, file=sys.stderr)
                elif os.path.isdir(path):
                    print('   %s is a directory' % apath, file=sys.stderr)
                elif stat.S_IXUSR & os.stat(path)[stat.ST_MODE]:
                    print('   %s is an executable' % apath, file=sys.stderr)
                else:
                    print('   %s cannot be blamed' % apath, file=sys.stderr)
                continue

            assert path_hash_loc[path], path
            assert sum(path_hash_loc[path].values()), path

            blamed += 1

        for path in set(file_list) - self.bad_files:
            if os.path.basename(path) in {'.gitignore'}:
                continue
            assert path in self.path_hash_loc, path

        print('~' * 80)
        duration = time.time() - start
        loc = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
        rate = blamed / duration if duration >= 1.0 else 0
        print('%d files,%d blamed,%d lines,%d commits,dt=%.1f,rate=%.1f' % (len(file_list), blamed,
              loc, len(hash_date_author), duration, rate))

        return len(path_set) > paths0

    def _check(self):
        # !@#$ remove
        if not STRICT_CHECKING:
            return
        for h_l in self.path_hash_loc.values():
            for hsh, loc in h_l.items():
                date, author = self.hash_date_author[hsh]

    def update_data(self, file_list, force):
        """Compute base statistics over whole revision
            blame all files in `path_patterns`
            Update: hash_date_author, path_hash_loc for files that are not already in path_hash_loc

            Also update exts_good, exts_good_loc, exts_bad, exts_ignored
        """
        n_paths0 = len(self.path_set)
        print('^' * 80)
        print('Update data for %d files' % len(file_list))
        self._check()
        if not force:
            self._update_from_existing(file_list)
            self._check()
        self._update_new_files(file_list, force)
        self._check()

        for path in set(file_list) - self.bad_files:
            assert path in self.path_hash_loc, path

        self._repo_map.catalog['hash_date_author'] = {hsh: (date, decode_str(author))
            for hsh, (date, author) in self._repo_map.catalog['hash_date_author'].items()}

        for path in set(file_list) - self.bad_files:
            assert path in self.path_hash_loc, path

        return len(self.path_set) > n_paths0


def __none_len(n, o):
    """Debugging code for checking args
    !@#$ Remove
    """
    assert o is None or isinstance(o, set), (n, type(o))
    if o is None:
        return None
    assert isinstance(o, (list, tuple, set)), o
    if len(o) <= 1:
        return o
    else:
        return (len(o), type(sorted(o)[0]))

    return None if o is None else (len(o), type(sorted(o)[0]))


def _filter_list(s_list, pattern):
    if pattern is None:
        return None
    regex = re.compile(pattern, re.IGNORECASE)
    return {s for s in s_list if regex.search(s)}


def filter_path_hash_loc(blame_map, path_hash_loc, file_list=None, author_list=None, ext_list=None):
    """Trim `path_hash_loc` down to files in `file_list` authors in `author_list` and extensions
        in `ext_list'
        Note: Does NOT modify path_hash_loc
        # TODO: inplace?
        !@#$ does hash_date_author need to be filtered?

    """

    hash_date_author = blame_map.hash_date_author
    path_hash_loc0 = path_hash_loc

    for path in path_hash_loc:
        assert path_hash_loc[path], path
        assert sum(path_hash_loc[path].values()), path

    all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

    if file_list is not None:
        file_list = set(file_list)
    if author_list is not None:
        author_list = set(author_list)

    if file_list or ext_list:
        path_hash_loc = {path: hash_loc for path, hash_loc in path_hash_loc.items()
                         if (not file_list or path in file_list) and
                            (not ext_list or get_ext(path) in ext_list)
                         }
    for path in path_hash_loc:
        assert path_hash_loc[path], path
        assert sum(path_hash_loc[path].values()), path

    if author_list:
        author_list = set(author_list)
        hash_set = {hsh for hsh, (_, author) in hash_date_author.items() if author in author_list}
        path_hash_loc = {path: {hsh: loc for hsh, loc in hash_loc.items() if hsh in hash_set}
                         for path, hash_loc in path_hash_loc.items()}
        path_hash_loc = {path: hash_loc for path, hash_loc in path_hash_loc.items() if hash_loc}

    all_authors = {a for _, a in blame_map.hash_date_author.values()}
    assert all_authors == all_authors0, (all_authors, all_authors0)

    # print('author_list:', author_list)
    for path in path_hash_loc:
        assert path_hash_loc[path], path
        assert sum(path_hash_loc[path].values()), path

    return path_hash_loc


class ReportMap(object):
    """ReportMaps contain data for reports. Unlike BlameMaps they don't make git calls, the only
        filter day, write reports and plot graphs so they _should_ be fast
    """

    def __init__(self, blame_map, path_hash_loc, path_patterns, reports_dir,
        file_list, author_pattern, ext_pattern, author_list=None):
        # assert isinstance(catalog, dict), type(catalog)
        assert author_pattern is None or isinstance(author_pattern, str), author_pattern
        assert ext_pattern is None or isinstance(ext_pattern, str), ext_pattern
        assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list

        # !@#$ Remove all the all_authors asserts
        all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

        for path in path_hash_loc:
            assert path_hash_loc[path], path
            assert sum(path_hash_loc[path].values()), path

        authors = {author for _, author in blame_map.hash_date_author.values()}
        exts = {get_ext(path) for path in path_hash_loc.keys()}
        self.author_list = _filter_list(authors, author_pattern)
        if not self.author_list:
            self.author_list = author_list
        elif author_list is not None:
            self.author_list &= set(author_list)

        self.ext_list = _filter_list(exts, ext_pattern)

        assert path_hash_loc
        # assert file_list
        if file_list:
            # print('!', len(file_list))
            # print('@', self.author_list)
            # print('@', self.ext_list)
            for path in file_list:
                assert path in path_hash_loc, path

        path_hash_loc = filter_path_hash_loc(blame_map, path_hash_loc, file_list, self.author_list,
                                             self.ext_list)

        assert path_hash_loc

        assert ':' not in reports_dir[2:], reports_dir
        self.reports_dir = reports_dir
        self.path_patterns = path_patterns

        all_authors = {a for _, a in blame_map.hash_date_author.values()}
        assert all_authors == all_authors0, (all_authors, all_authors0)
        self.path_hash_loc = path_hash_loc


def decode_str(s):
    if s is None:
        return s
    try:
        return s.decode('utf-8')
    except:
        return s.decode('latin-1')


def derive_blame(path_hash_loc, hash_date_author):
    """Compute summary tables over whole repository:hash
      !@#$ Either filter this to report or compute while blaming
      or limit to top authors <=== by trimming path_hash_loc
      !@#$ detailed: list of
    """

    # hash_date_author = {hsh: (date, decode_str(author))
    #                     for hsh, (date, author) in hash_date_author.items()}

    exts = {get_ext(path) for path in path_hash_loc.keys()}
    authors = {decode_str(author) for _, author in hash_date_author.values()}

    authors = sorted(authors)

    df_ext_author_files = DataFrame(index=exts, columns=authors)
    df_ext_author_loc = DataFrame(index=exts, columns=authors)
    df_ext_author_files.iloc[:, :] = df_ext_author_loc.iloc[:, :] = 0

    print('derive_blame: exts=%d,authors=%d,product=%d, path_hash_loc=%d files, %d lines' % (
          len(exts), len(authors),
          len(exts) * len(authors),
          len(path_hash_loc),
          sum(sum(v.values()) for v in path_hash_loc.values())
          ))

    for path, v in path_hash_loc.items():
        assert sum(v.values()), (path, len(v))
        # print('#$', len(v), sum(v.values()), path)

    for path, hash_loc in path_hash_loc.items():
        ext = get_ext(path)
        for hsh, loc in hash_loc.items():
            _, author = hash_date_author[hsh]
            df_ext_author_files.loc[ext, author] += 1
            df_ext_author_loc.loc[ext, author] += loc

    return df_ext_author_files, df_ext_author_loc


def get_tree_loc(path_loc):

    dir_tree = defaultdict(set)
    roots = set()

    for path in path_loc.keys():
        child = path
        while True:
            parent = os.path.dirname(child)
            if parent == child:
                roots.add(parent)
                break
            dir_tree[parent].add(child)
            child = parent

    dir_tree = {path: sorted(dir_tree[path]) for path in sorted(dir_tree.keys())}

    dir_loc = Counter()
    stack = []
    for root in roots:
        stack.append((root, 0))
        while stack:
            parent, i = stack[-1]
            stack[-1] = parent, i + 1
            if parent not in dir_tree:
                # Terminal node
                dir_loc[parent] += path_loc[parent]
                stack.pop()
            else:
                if i < len(dir_tree[parent]):
                    stack.append((dir_tree[parent][i], 0))
                else:
                    dir_loc[parent] = path_loc.get(parent, 0) +\
                                      sum(dir_loc[child] for child in dir_tree[parent])
                    stack.pop()

    dir_loc = {d: l for d, l in dir_loc.items()}
    pure_dir_loc = {d: l for d, l in dir_loc.items() if d in dir_tree and l}

    pure_dir_frac = {d: (l, 0) for d, l in pure_dir_loc.items()}
    for d, l in pure_dir_loc.items():
        for child in dir_tree[d]:
            if child not in pure_dir_loc:
                continue
            pure_dir_frac[child] = tuple((dir_loc[child], dir_loc[child] / l))

    return pure_dir_frac


def detailed_loc(path_hash_loc, reports_dir):

    path_loc = {path: sum(loc for _, loc in hash_loc.items())
                for path, hash_loc in path_hash_loc.items()}

    dir_tree_loc = get_tree_loc(path_loc)
    dir_loc_frac = [(d, l, f) for d, (l, f) in dir_tree_loc.items()]
    dir_loc_frac.sort()

    return DataFrame(dir_loc_frac, columns=['dir', 'LoC', 'frac'])


def save_tables(blame_map, reports_map, summary, detailed):
    hash_date_author = blame_map.hash_date_author
    path_hash_loc = reports_map.path_hash_loc
    reports_dir = reports_map.reports_dir

    if not path_hash_loc:
        print('No files to process')
        return False
    if not (summary or detailed):
        print('Nothing to do')
        return False

    df_ext_author_files, df_ext_author_loc = derive_blame(path_hash_loc, hash_date_author)

    def make_path(key):
        return os.path.join(reports_dir, '%s.csv' % key)

    mkdir(reports_dir)
    if summary:
        df_append_totals(df_ext_author_files).to_csv(make_path('ext_author_files'))
        df_append_totals(df_ext_author_loc).to_csv(make_path('ext_author_loc'))

    if detailed:
        df_dir_tree_loc = detailed_loc(path_hash_loc, reports_dir)
        df_dir_tree_loc.to_csv(make_path('details'))
        print('Details: %s' % make_path('details'))

    return True


DATE_INF_NEG = Timestamp('1911-11-22 11:11:11 -0700')
DATE_INF_POS = Timestamp('2111-11-22 11:11:11 -0700')


def get_top_authors(blame_map, report_map):
    """Get top authors in `report_map`
    """

    hash_date_author = blame_map.hash_date_author
    path_hash_loc = report_map.path_hash_loc

    # author_loc_dates = {author: loc, min date, max date} over all hashes
    author_loc_dates = defaultdict(lambda: [0, DATE_INF_POS, DATE_INF_NEG])
    i = 0
    for hash_loc in path_hash_loc.values():
        for hsh, loc in hash_loc.items():
            date, author = hash_date_author[hsh]
            loc_dates = author_loc_dates[author]
            loc_dates[0] += loc
            loc_dates[1] = min(loc_dates[1], date)
            loc_dates[2] = max(loc_dates[2], date)
            i += 1

    assert author_loc_dates
    return author_loc_dates, sorted(author_loc_dates.keys(), key=lambda a: -author_loc_dates[a][0])


def analyze_blame(blame_map, report_map):
    """TODO: Add filter by extensions and authors
    """
    hash_date_author = blame_map.hash_date_author
    path_hash_loc = report_map.path_hash_loc

    # dates = dates of all commits
    dates = [date for date, _ in hash_date_author.values()]

    # hash_loc = {hsh: loc} over all hashes
    hash_loc = Counter()
    for path, h_l in path_hash_loc.items():
        for hsh, loc in h_l.items():
            hash_loc[hsh] += loc

    # Populate the following dicts
    author_loc = Counter()              # {author: loc}
    author_dates = {}                   # {author: (min date, max date)}
    author_date_hash_loc = {}           # {author: [(date, hsh, loc)]}

    assert hash_loc and hash_date_author

    # hash_loc can be very big, e.g. 200,000 for linux source

    for i, (hsh, loc) in enumerate(hash_loc.items()):
        date, author = hash_date_author[hsh]
        author_loc[author] += loc
        if author not in author_dates:
            author_dates[author] = [date, date]
        else:
            date_min, date_max = author_dates[author]
            author_dates[author] = [min(date, date_min), max(date, date_max)]

        if author not in author_date_hash_loc.keys():
            author_date_hash_loc[author] = []
        author_date_hash_loc[author].append((date, hsh, loc))

    assert author_loc

    for author in author_date_hash_loc.keys():
        author_date_hash_loc[author].sort()

    # author_stats = {author: (loc, (min date, max date), #days, ratio)}
    author_stats = {}
    for author in sorted(author_loc.keys(), key=lambda k: -author_loc[k]):
        loc = author_loc[author]
        dates = author_dates[author]
        days = (dates[1] - dates[0]).days
        ratio = loc / days if days else 0.0
        author_stats[author] = (loc, dates, days, ratio)

    author_date_loc = defaultdict(list)
    for hsh, loc in hash_loc.items():
        date, author = hash_date_author[hsh]
        author_date_loc[author].append((date, loc))

    return {'author_list': report_map.author_list,
            'ext_list': report_map.ext_list,
            'hash_date_author': hash_date_author,
            'hash_loc': hash_loc,
            'author_date_loc': author_date_loc,
            'author_date_hash_loc': author_date_hash_loc,
            'author_stats': author_stats}


def get_peak_commits(hash_loc, date_hash_loc, history, window=20 * DAY):
    """Return lists of commits around peaks in a time series
    """
    ts, peak_ixy = history
    dt = window / 2

    peak_ixy = sorted(peak_ixy, key=lambda k: key_ixy_x(*k))

    peak_ends = [[x - dt, x + dt] for _, x, _ in peak_ixy]
    for i in range(1, len(peak_ends)):
        m0, m1 = peak_ends[i - 1]
        n0, n1 = peak_ends[i]
        if m1 > n0:
            peak_ends[i - 1][1] = peak_ends[i][0] = m1 + (n0 - m1) / 2

    peak_commits = []
    for (i, x, y), (m0, m1) in zip(peak_ixy, peak_ends):
        assert isinstance(x, pd.Timestamp), (type(x), x)
        mode_hashes = [hsh for (date, hsh, loc) in date_hash_loc if m0 <= date < m1]
        mode_hashes.sort(key=lambda hsh: -hash_loc[hsh])
        loc = sum(hash_loc[hsh]for hsh in mode_hashes)
        peak_commits.append((loc, x, mode_hashes))
    loc_total = sum(loc for loc, _, _ in peak_commits)

    return loc_total, list(zip(peak_ixy, peak_commits))


def plot_analysis(blame_map, report_map, history, author, do_show, graph_path):

    # !@#$ update figure number
    fig, ax0 = plt.subplots(nrows=1)

    label = None
    ts, peak_idx = history

    plot_loc_date(ax0, label, (ts, peak_idx))
    plot_show(ax0, blame_map, report_map, author, do_show, graph_path, False)


def aggregate_author_date_hash_loc(author_date_hash_loc, author_list=None):
    if author_list is None:
        author_list = author_date_hash_loc.keys()
    date_hash_loc = []
    for author in author_list:
        date_hash_loc.extend(author_date_hash_loc[author])
    return date_hash_loc


def aggregate_author_stats(author_stats, author_list):
    assert author_stats
    assert author_list
    a_loc, (a_date_min, a_date_max), _, _ = author_stats[author_list[0]]
    for author in author_list[1:]:
        loc, (date_min, date_max), _, _ = author_stats[author]
        a_loc += loc
        a_date_min = min(a_date_min, date_min)
        a_date_max = max(a_date_max, date_max)
    a_days = delta_days(a_date_min, a_date_max)
    a_ratio = a_loc / a_days if a_days else 0
    return a_loc, (a_date_min, a_date_max), a_days, a_ratio


def write_legend(legend_path, author_date_loc, hash_loc, history, date_hash_loc,
    hash_date_author, author, n_top_commits):

    loc_auth, peak_ixy_commits = get_peak_commits(hash_loc, date_hash_loc, history)
    peak_ixy_commits.sort(key=lambda k: key_ixy_x(*k[0]))

    with open(legend_path, 'wt') as f:

        def put(s):
            f.write('%s\n' % s.encode('utf-8'))

        put('=' * 80)
        put('%s: %d peaks %d LoC' % (author, len(peak_ixy_commits), loc_auth))

        for (i, x, y), (loc, peak, mode_hashes) in peak_ixy_commits:
            put('.' * 80)
            put('%3d) %d commits %d LoC around %s' % (i + 1, len(mode_hashes), loc, date_str(peak)))
            for hsh in sorted(mode_hashes[:n_top_commits], key=lambda k: hash_date_author[k][0]):
                put('%5d LoC, %s %s' % (hash_loc[hsh], date_str(hash_date_author[hsh][0]),
                    git_show_oneline(hsh)))


def put_newest_oldest(f, author, hash_path_loc, date_hash_loc, hash_date_author, n_revisions,
    n_files, is_newest):
    """Write a report of oldest surviving revisions in current revision
        n_revisions: Max number of revisions to write
        n_files: Max number of files to write for each revision

        author: Date of oldest revision
        revision 1: Same format as write_legend
            file 1: LoC
            file 2: LoC
            ...
        revision 2: ...
            ...
    """
    def put(s):
        f.write('%s\n' % s)

    def date_key(date, hsh, loc):
        return date, loc

    date_hash_loc = sorted(date_hash_loc, key=lambda k: date_key(*k))
    if is_newest:
        date_hash_loc.reverse()
    loc_total = sum(loc for _, _, loc in date_hash_loc)

    put('=' * 80)
    put('%s: %d commits %d LoC' % (author, len(date_hash_loc), loc_total))

    for i, (date, hsh, loc) in enumerate(date_hash_loc[:n_revisions]):
        put('.' * 80)
        put('%5d LoC, %s %s' % (loc, date_str(hash_date_author[hsh][0]),
            git_show_oneline(hsh)))

        path_loc = sorted(hash_path_loc[hsh].items(), key=lambda k: (-k[1], k[0]))
        for j, (path, l) in enumerate(path_loc[:n_files]):
            put('%5d LoC,   %s' % (l, path))


def write_newest(newest_path, author, hash_path_loc, date_hash_loc, hash_date_author, n_revisions,
    n_files):

    with open(newest_path, 'wt') as f:
        put_newest_oldest(f, author, hash_path_loc, date_hash_loc, hash_date_author, n_revisions,
                          n_files, True)


def write_oldest(oldest_path, author, hash_path_loc, date_hash_loc, hash_date_author, n_revisions,
    n_files):

    with open(oldest_path, 'wt') as f:
        put_newest_oldest(f, author, hash_path_loc, date_hash_loc, hash_date_author, n_revisions,
                          n_files, False)


def make_hash_path_loc(path_hash_loc):
    # print('make_hash_path_loc 1: path_hash_loc=%d' % len(path_hash_loc))
    hash_path_loc = defaultdict(dict)
    for path, hash_loc in path_hash_loc.items():
        for hsh, loc in hash_loc.items():
            hash_path_loc[hsh][path] = loc
    # print('make_hash_path_loc 2: hash_path_loc=%d' % len(hash_path_loc))
    return hash_path_loc


def save_analysis(blame_map, report_map, analysis, _a_list, do_save, do_show,
    _n_top_authors, n_peaks, n_top_commits, n_oldest, n_files, n_min_days):
    """Create a graph (time series + markers)
        a list of commits in for each peak
        + n biggest commits
        <name>.png
        <name>.txt
    """
    reports_dir = report_map.reports_dir

    mkdir(reports_dir)

    hash_date_author = analysis['hash_date_author']
    hash_loc = analysis['hash_loc']
    _author_date_hash_loc = analysis['author_date_hash_loc']
    author_date_loc = analysis['author_date_loc']
    ext_list = analysis['ext_list']

    if not _a_list:
        author = '[all]'
    elif len(_a_list) == 1:
        author = _a_list[0]
    else:
        assert False, _a_list
    date_hash_loc = aggregate_author_date_hash_loc(_author_date_hash_loc, _a_list)
    history = make_history(author_date_loc, n_peaks, _a_list)
    tsm, _ = history
    if tsm.max() - tsm.min() < n_min_days:
        print('%d days of activity < %d. Not reporting' % (tsm.max() - tsm.min(), n_min_days))
        return False

    # !@#$ need a better name than history
    report_name = '[%s]' % (concat(sorted(ext_list)) if ext_list else 'all')
    if do_save:
        graph_name = 'code-age%s.png' % report_name
        legend_name = 'code-age%s.txt' % report_name
        newest_name = 'newest%s.txt' % report_name
        oldest_name = 'oldest%s.txt' % report_name
        graph_path = os.path.join(reports_dir, graph_name)
        legend_path = os.path.join(reports_dir, legend_name)
        newest_path = os.path.join(reports_dir, newest_name)
        oldest_path = os.path.join(reports_dir, oldest_name)

        hash_path_loc = make_hash_path_loc(blame_map.path_hash_loc)
        write_legend(legend_path, author_date_loc, hash_loc, history, date_hash_loc,
                     hash_date_author, author, n_top_commits)
        write_newest(newest_path, author, hash_path_loc, date_hash_loc, hash_date_author,
                     n_oldest, n_files)
        write_oldest(oldest_path, author, hash_path_loc, date_hash_loc, hash_date_author,
                     n_oldest, n_files)
    else:
        graph_path = None

    # author = 'all' if not a_list else a_list[0]
    plot_analysis(blame_map, report_map, history, author, do_show, graph_path)
    return True


def create_reports(gitstatsignore, path_patterns, do_save, do_show, force,
   author_pattern, ext_pattern,
   n_top_authors, n_peaks, n_top_commits, n_oldest, n_files, n_min_days):

    remote_url, remote_name = git_remote()
    description = git_describe()
    revision_hash = git_current_revision()
    date = git_date(revision_hash)
    date_ymd = date.strftime('%Y-%m-%d')

    # git.stats directory layout
    # root\             ~/git.stats/
    #   repo\           ~/git.stats/papercut
    #     rev\          ~/git.stats/papercut/2015-11-16.3f4632c6
    #       reports\    ~/git.stats/papercut/2015-11-16.3f4632c6/reports/tools_page-analysis-tools
    root_dir = os.path.join(os.path.expanduser('~'), 'git.stats')
    repo_base_dir = os.path.join(root_dir, remote_name)
    repo_dir = os.path.join(repo_base_dir, 'reports')
    rev_dir = os.path.join(repo_dir, '.'.join([date_ymd, truncate_hash(revision_hash),
                                               clean_path(description)]))

    path_patterns = [normalize_path(path) for path in path_patterns]
    if not path_patterns or (len(path_patterns) == 1 and path_patterns[0] == '.'):
        reports_name = '[root]'
    else:
        reports_name = '.'.join(clean_path(path) for path in path_patterns)
    reports_dir = os.path.join(rev_dir, reports_name)[:PATH_MAX - 50]

    # !@#$ TODO Add a branches file in rev_dir
    repo_summary = {
        'remote_url': remote_url,
        'remote_name': remote_name,
    }
    rev_summary = {
        'revision_hash': revision_hash,
        'branch': git_current_branch(),
        'description': description,
        'name': git_name(),
        'date': date,
    }

    date = max(MAX_DATE, date)  # !@#$ Hack, Needed because merged code can be newer than branch

    all_summary = repo_summary.copy()
    all_summary.update(rev_summary)

    print('=' * 80)
    for k, v in sorted(all_summary.items()):
        print('%15s: %s' % (k, v))
    print('-' * 80)

    ignored_files = get_ignored_files(gitstatsignore)

    file_list0 = git_file_list(path_patterns)
    file_list = {path for path in file_list0
                 if get_ext(path) not in IGNORED_EXTS}
    file_list -= ignored_files
    print('path_patterns=%s' % path_patterns)
    print('file_list=%d raw, %d filtered' % (len(file_list0), len(file_list)))

    if not file_list:
        print('path_patterns=%s selects no files. Nothing to do' % path_patterns)
        return

    blame_map = BlameMap(repo_base_dir, repo_summary, rev_summary)

    if not force:
        blame_map.load()

    changed = blame_map.update_data(file_list, force)

    for path in blame_map.path_hash_loc:
        assert blame_map.path_hash_loc[path], os.path.abspath(path)
        assert sum(blame_map.path_hash_loc[path].values()), os.path.abspath(path)

    if changed:
        blame_map.save(force)

    for path in blame_map.path_hash_loc:
        assert blame_map.path_hash_loc[path], os.path.abspath(path)
        assert sum(blame_map.path_hash_loc[path].values()), os.path.abspath(path)

    file_set = set(file_list)
    if blame_map.bad_files:
        file_set2 = file_set - blame_map.bad_files
        print('`' * 80)
        print('%d bad files' % len(blame_map.bad_files & file_set))
        for i, path in enumerate(sorted(blame_map.bad_files & file_set)):
            print('%3d: %s' % (i, path))
        print('"' * 80)

        print('file_list=%d (%d bad)' % (len(file_set2), len(blame_map.bad_files & file_set)))
        file_set = file_set2

    if not file_set:
        print('Only bad files. Nothing to do' % path_patterns)
        return

    report_map_all = ReportMap(blame_map, blame_map.path_hash_loc, path_patterns, reports_dir,
                               file_set, author_pattern, ext_pattern)

    author_loc_dates, top_authors = get_top_authors(blame_map, report_map_all)

    top_a_list = [None] + [[a] for a in top_authors[:n_top_authors]]
    reports_dir_list = []
    # all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

    save_tables(blame_map, report_map_all, summary=True, detailed=True)

    for a_list in top_a_list:
        # all_authors = {a for _, a in blame_map.hash_date_author.values()}
        reports_dir_author = '[all]' if a_list is None else a_list[0]
        reports_dir_author = os.path.join(reports_dir, clean_path(reports_dir_author))
        report_map = ReportMap(blame_map, report_map_all.path_hash_loc, path_patterns,
                               reports_dir_author,
                               None, None, None, a_list)

        analysis = analyze_blame(blame_map, report_map)

        if not save_analysis(blame_map, report_map, analysis, a_list, do_save, do_show,
                             n_top_authors, n_peaks, n_top_commits, n_oldest, n_files, n_min_days):
            continue
        save_tables(blame_map, report_map, summary=False, detailed=True)
        reports_dir_list.append(os.path.abspath(report_map.reports_dir))

    print('+' * 80)
    print('rev_dir=%s' % os.path.abspath(rev_dir))
    for reports_dir_author in reports_dir_list:
        print('reports_dir=%s' % reports_dir_author)

    print('description="%s"' % blame_map.get_description())


if __name__ == '__main__':

    lowpriority()

    import optparse
    parser = optparse.OptionParser('python ' + sys.argv[0] + ' [options] [<directory>]')
    parser.add_option('-c', '--code-only', action='store_true', default=False, dest='code_only',
                      help='Show only code files')
    parser.add_option('-f', '--force', action='store_true', default=False, dest='force',
                      help='Force running git blame over source code')
    parser.add_option('-s', '--show', action='store_true', default=False, dest='do_show',
                      help='Pop up graphs as we go')
    parser.add_option('-a', '--authors', dest='author_pattern', default=None,
                      help='Analyze only code with these authors')
    parser.add_option('-e', '--extensions', dest='ext_pattern', default=None,
                      help='Analyze only files with these extensions')
    parser.add_option('-g', '--gitstatsignore', dest='gitstatsignore', default=None,
                      help='File patterns to ignore')

    # Display / Report options
    parser.add_option('-A', '--number-top-authors', dest='n_top_authors', type=int, default=20,
                      help='Number of authors to list')
    parser.add_option('-P', '--number-peaks', dest='n_peaks', type=int, default=10,
                      help='Number of peaks to find in a code age graph')
    parser.add_option('-O', '--number-oldest-tweets', dest='n_oldest', type=int, default=20,
                      help='Number of oldest (and newest) commits to list')
    parser.add_option('-C', '--number-commits', dest='n_top_commits', type=int, default=5,
                      help='Number of commits to list for each author')
    parser.add_option('-F', '--number-files', dest='n_files', type=int, default=3,
                      help='Number of files to list for each commit in a report')
    parser.add_option('-D', '--min-days', dest='n_min_days', type=int, default=1,
                      help='Minimum days of blamed code to report')

    do_save = True

    options, args = parser.parse_args()

    create_reports(options.gitstatsignore,
                   args, do_save, options.do_show, options.force,
                   options.author_pattern, options.ext_pattern,
                   options.n_top_authors, options.n_peaks,
                   options.n_top_commits, options.n_oldest, options.n_files,
                   options.n_min_days)
