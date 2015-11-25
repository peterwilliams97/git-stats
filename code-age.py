from __future__ import division, print_function
"""
    Analyze code age in a git repository

    Key Data Structures
    -------------------
    manifest_dict: Full manifest of data files used by this script
    hash_loc: {hsh: loc} over all hashes
    hash_date_author: {hsh: (date, author)} over all hashes  !@#$ Combine with hash_loc
    author_date_hash_loc: {author: [(date, hsh, loc)]} over all authors.
                          [(date, hsh, loc)]  over all author's commits
    author_stats: {author: (loc, (min date, max date), #days, ratio)} over authors
                   ratio = loc / #days
    author_history: {author: (ts, peaks)} over all authors
    author_list: sorted list of authors. !@#$ Not needed. Sort with key based on author_stats

    References
    ----------
    git log --pretty=oneline -S'PDL'
    http://stackoverflow.com/questions/8435343/retrieve-the-commit-log-for-a-specific-line-in-a-file

    http://stackoverflow.com/questions/4082126/git-log-of-a-single-revision
    git log -1 -U


"""
import subprocess
from collections import defaultdict, Counter
import sys
import time
import re
import os
import errno
import cPickle as pickle
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Timestamp
import matplotlib
import matplotlib.pylab as plt
from pprint import pprint
import copy


#
# Configuration
#
TIMEZONE = 'Australia/Melbourne'
HASH_LEN = 8
N_PEAKS = 50


# Set graphing style
matplotlib.style.use('ggplot')
plt.rcParams['axes.color_cycle'] = ['b', 'y', 'k', '#707040', '#404070'] + plt.rcParams['axes.color_cycle'][1:]

print(plt.rcParams['savefig.dpi'])
plt.rcParams['savefig.dpi'] = 300
print(plt.rcParams['savefig.dpi'])


IGNORED_EXTS = {
   '.exe', '.png', '.bmp', '.pfx', '.gif', '.jpg', '.svg', '.jpeg',
   '.spc', '.developerprofile', '.cer', '.der', '.cert', '.keychain', '.pem',
   '.icns', '.ico', '.bin', '.xls', '.xlsx', '.prn', '.launch', '.swf', '.air',
   '.jar', '.so', '.doc',
   # '.properties'
   }

# TODO: Try to follow .gitignore patterns
IGNORED_PATTERNS = [
    'pre-built', '/bin/', '/JavaApplicationStub', '/JavaAppLauncher',
    'providers/print/lib/'
    ]

GUILD_EXT = {
    'java': {'.java'},
    'c': {'.c', '.h', '.cpp', '.cs'},
    'web': {'.htm', '.html', '.js', '.page', '.jsp', '.css'},
    'strings': {'.properties', },
    'script': {'.py', '.pl', '.sh', '.bat', '.cmd', '.transform'},
    'basic': {'.bas', '.vbs'},
    'build': {'.sln', '.vcproj', '.csproj''.gradle',}
}

EXT_GUILD = {ext: gld for gld, extensions in GUILD_EXT.items() for ext in extensions}


def show_versions():
    print('python    : %s.%s.%s' % sys.version_info[:3])
    print('numpy     : %s' % np.__version__)
    print('matplotlib: %s' % matplotlib.__version__)
    print('pandas    : %s '% pd.__version__)


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
        import win32api, win32process, win32con

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


def to_timestamp(date_s):
    """Convert string `date_s' to pandas Timestamp in `TIMEZONE`
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
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    assert os.path.exists(path), path

    # Delete old if load_object succeeds
    load_object(path)
    if os.path.exists(path) and os.path.exists(existing_pkl):
        os.remove(existing_pkl)


def load_object(path, default=None):
    """Load object from `path`"""
    if default is not None and not os.path.exists(path):
        return default
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        print('load_object(%s, %s) failed' % (path, default), file=sys.stderr)
        raise


def mkdir(path):
    """Create directory dir and ignore already exists errors
    """
    print('mkdir: path="%s"' % path)  # !@#$ remove
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise
    assert os.path.exists(path), path  # !@#$ now superfluous


def df_append_totals(df_in):
    """Append row and column totals to Pandas DataFrame `df_in` and sort rows and columns by total
        Returned data frame has +1 rows and columns compared to input data fram
        TODO: Option to insert at end
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
    return df


TOLERANCE = 1e-6

def moving_average(series, window):
    """Return weighted moving average of Pandas Series `series`
        Weights are a triangle of width `window`
    """
    n = len(series)
    d0 = (window) // 2
    d1 = window - d0
    weights = np.empty(window, dtype=np.float)
    radius = (window - 1) / 2
    for i in xrange(window):
        weights[i] = radius + 1 - abs(i - radius)

    n = len(series)
    ma = np.empty(n, dtype=float)
    for i in xrange(n):
        i0 = max(0, i - d0)
        i1 = min(n, i + d1)
        c0 = i0 - (i - d0)
        c1 = (i + d1) - n

        v = np.average(series[i0:i1], weights=weights[c0:window - c1])
        ma[i] = min(max(series[i0:i1].min(), v), series[i0:i1].max())

    return Series(ma, index=series.index)


def printable(s):
    assert isinstance(s, str), (type(s), s)
    r = repr(s)
    if len(r) > 100:
        return '%s ... %s' % (r[:75], r[-20:])
    return r


def get_ext(path):
    parts = os.path.splitext(path)
    return parts[-1] if parts else '[None]'


# TODO strip leading / trailing space
# TODO save stderr and print it on error
def exec_output(command, require_output):
    output = subprocess.check_output(command)
    if require_output and not output:
        raise RuntimeError('exec_output: command=%s' % command)
    return output


def exec_output_lines(command, require_output):
    return exec_output(command, require_output).splitlines()


def exec_headline(command):
    return exec_output(command, True).splitlines()[0]


def git_file_list(path_list=()):
    return exec_output_lines(['git', 'ls-files'] + path_list, False)


def git_show_oneline(obj):
    """https://git-scm.com/docs/git-show
    """
    return exec_headline(['git', 'show', '--oneline', '--quiet', obj])


def git_date(obj):
    date_s = exec_headline(['git', 'show', '--pretty=format:%ai', '--quiet', obj])
    return to_timestamp(date_s)


RE_REMOTE_URL =  re.compile(r'(https?://.*/[^/]+(?:\.git)?)\s+\(fetch\)')
RE_REMOTE_NAME = re.compile(r'https?://.*/(.+?)(\.git)?$')


def git_remote():
    """Returns: remote_url, remote_name
    """
    # $ git remote -v
    # origin  https://github.com/FFTW/fftw3.git (fetch)
    # origin  https://github.com/FFTW/fftw3.git (push)

    for ln in exec_output_lines(['git', 'remote', '-v'], True):
        m = RE_REMOTE_URL.search(ln)
        if not m:
            continue
        remote_url = m.group(1)
        remote_name = RE_REMOTE_NAME.search(remote_url).group(1)
        return remote_url, remote_name

    raise RuntimeError('No remote')


def git_describe():
    return exec_headline(['git', 'describe'])


def git_name():
    """Returns name of current commit
    """
    return ' '.join(exec_headline(['git', 'name-rev', 'HEAD']).split()[1:])


def git_current_branch():
    branch = exec_headline(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == 'HEAD': # Detached HEAD?
        branch = '[None]'
    return branch


def git_current_commit():
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
    # https://coderwall.com/p/x8xbnq/git-don-t-blame-people-for-changing-whitespaces-or-moving-code
    # -l for long hashes
    assert os.path.exists(path), path
    try:
        return exec_output(['git', 'blame', '-l', '-f', '-w', '-M', path], False)
    except Exception as e:
        # !@#$ Get stderr from exec_output
        print('git_blame_text:\n\tpath=%s\n\te=%s' % (path, e), file=sys.stderr)
        raise

RE_BLAME = re.compile(r'''
    \^*([0-9a-f]{4,})\s+
    .+?\s+
    \(
    (.+?)\s+
    (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})
    \s+(\d+)
    \)''',
    re.DOTALL | re.MULTILINE | re.VERBOSE)
if False:
    text = r'7aa60cb5 (Peter Williams 2013-09-03 04:13:46 +0000   1) #! /bin/sh'
    m = RE_BLAME.search(text)
    print(m.groups())
    exit()

MAX_DATE = Timestamp('2015-12-22 14:11:54 -0700') # !@#$ Now


def get_text_hash_loc(hash_date_author, path_hash_loc, path_set, author_set, ext_set,
    text, path, max_date=None):
    """Update key data structures with information that we parse from `text` in this function
    """

    if max_date is None:
        max_date = MAX_DATE
    else:
        assert max_date < MAX_DATE, max_date

    hash_loc = Counter()
    for i, ln in enumerate(text.splitlines()):
        m = RE_BLAME.match(ln)
        if not m:
            print('    Bad file: "%s"' % path)
            return None
        assert m, (path, i, len(ln), ln[:200])
        if m.group(2) == 'Not Committed Yet':
            continue
        hsh = m.group(1)
        author = m.group(2)
        date_s = m.group(3)
        line_n = int(m.group(4))
        if line_n != i + 1:
            return None

        author = author.strip()
        if author == '':
            author = '<>'

        assert line_n == i + 1, 'line_n=%d,i=%d\n%s\n%s' % (line_n, i, path, m.group(0))
        assert author.strip() == author, 'author="%s\n%s:%d\n%s",' % (author, path, i + 1, ln[:200])

        date = Timestamp(date_s)
        date = date.tz_convert('Australia/Melbourne')
        # assert date <= max_date

        if date > max_date:
            print('Bogus timestamp: %s:%d\n%s' % (path, i + 1, ln[:200]))
            continue

        hash_loc[hsh] += 1
        hash_date_author[hsh] = (date, author)
        author_set.add(author)

    path_hash_loc[path] = hash_loc


def derive_blame(path_hash_loc, hash_date_author, exts_good):
    """Compute summary tables over whole repository:hash
      !@#$ Either filter this to report or compute while blaming
      or limit to top authors <=== by trimming path_hash_loc
    """
    authors = sorted(set(author for _, author in hash_date_author.values()))
    exts = sorted(exts_good.keys(), key=lambda k: -exts_good[k])
    df_ext_author_files = DataFrame(index=exts, columns=authors)
    df_ext_author_loc = DataFrame(index=exts, columns=authors)
    df_ext_author_files.iloc[:, :] = df_ext_author_loc.iloc[:, :] = 0

    print('derive_blame: path_hash_loc=%d, %d' % (
         len(path_hash_loc),
         sum(len(v) for v in path_hash_loc.values())))
    for path, hash_loc in path_hash_loc.items():
        ext = get_ext(path)
        for hsh, loc in hash_loc.items():
            _, author = hash_date_author[hsh]
            df_ext_author_files.loc[ext, author] += 1
            df_ext_author_loc.loc[ext, author] += loc

    print('derive_blame: df_ext_author_files=%d,df_ext_author_loc=%d' % (len(df_ext_author_files),
          len(df_ext_author_loc)))
    return df_ext_author_files, df_ext_author_loc


def ext_to_guild(df_author_ext):
    guilds = sorted({EXT_GUILD.get(ext, 'UNKN0WN') for ext in df_author_ext.index})
    df_author_guild = DataFrame(index=guilds, columns=df_author_ext.columns)
    for gld in guilds:
        extensions = sorted({ext for ext in df_author_ext.index if EXT_GUILD.get(ext, 'UNKN0WN') == gld})
        for ext in extensions:
            assert ext in df_author_ext.index, (ext, df_author_ext.index)
        df_author_guild.loc[gld, :] = df_author_ext.loc[extensions, :].sum(axis=0)
    return df_author_guild


def print_counts(title, counter):
    """Debug code for counters !@#$
    """
    if counter:
        max_klen = max(len(k) for k in counter.keys())
        max_vlen = max(len(str(v)) for v in counter.values())
        # print('!@', max_klen, max_vlen)
    total = sum(counter.values())

    print('-' * 80)
    print('%s : number=%d,total=%d' % (title, len(counter), total))
    if counter:
        for name, count in counter.most_common():
            print('%s %s %4.1f%%' % (name.ljust(max_klen), str(count).rjust(max_vlen),
                  count / total * 100.0))


def ts_summary(ts, desc=None):
    """Return a summary of time series `ts`
    """
    if len(ts) == 1:
        return '[singleton]'

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

    MIN_PEAK_DAYS = 60
    MAX_PEAK_DAYS = 1
    # !@#$ Tune np.arange(2, 10) * 10 to data
    peak_idx = signal.find_peaks_cwt(ts, np.arange(2, 10) * 10)
    peak_idx.sort(key=lambda k: -ts.iloc[k])
    peak_idx = [i for i in peak_idx
                if (delta_days(ts.index[0], ts.index[i]) >= MIN_PEAK_DAYS and
                    delta_days(ts.index[i], ts.index[-1]) >= MAX_PEAK_DAYS)
                ]
    return peak_idx

#
# Time series analysis
#  !@#$ try different windows to get better peaks
def __analyze_time_series(loc, date, window=60):
    """Return a histogram of LoC / day for events given by `loc` and `date`
        loc: list of LoC events
        date: list of timestamps for loc
        window: width of weighted moving average window used to smooth data
        Returns: averaged time series, list of peaks in time series
    """
    # ts is histogram of LoC with 1 day bins. bins start at midnight on TIMEZONE
    # !@#$ maybe offset dates in ts to center of bin (midday)

    ts = Series(loc, index=date) # !@#$ dates may not be unique, guarantee this
    ts_days = ts.resample('D', how='mean')  # Day
    ts_days = ts_days.fillna(0)

    # tsm is smoothed ts
    ts_ma = moving_average(ts_days, window) if window else ts_days

    peak_idx = find_peaks(ts_ma)

    return ts_ma, peak_idx


def make_history(author_date_loc, author_list=None):
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
    ts, peak_idx = __analyze_time_series(loc, date)
    # print('make_history: ts=%s' % ts_summary(ts, author_list))

    assert ts.index.max() <= MAX_DATE

    peak_pxy = [(p, ts.index[p], ts.iloc[p]) for p in peak_idx]

    def key_pxy(p, x, y):
        return -y, x, p

    peak_pxy.sort(key=lambda k: key_pxy(*k))
    peak_ixy = [(i, x, y) for i, (p, x, y) in enumerate(peak_pxy)]

    return ts, tuple(peak_ixy)


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
                               if yy > y - txt_height
                               and abs(xx - x) < txt_width * 2
                               and (yy, xx) != (y, x))
        if not yx_collisions or abs(yx_collisions[0][0] - y) >= txt_height:
            continue

        # yx_plot[i] is colliding with yx_collisions[0], the lowest colliding point

        # Search for lowest space between collisions that is big enough for text
        for j, (dy, dx) in enumerate(np.diff(yx_collisions, axis=0)):
            if dy > txt_height * 2: # Room for text?
                yx_plot[i][0] = yx_collisions[j][0] + txt_height
                break
        else:
            # move yx_plot[i] above yx_collisions[-1], the highest colliding point
            yx_plot[i][0] = yx_collisions[-1][0] + txt_height

    # Move points down to any gaps that have opened in above moves

    # try 5 times to give points a chance to settle
    for n in xrange(100):
        n_changes = 0
        changed = False
        yx_plot1 = [(y, x) for y, x in yx_plot]

        for i, ((y0, x0), (y, x)) in enumerate(zip(yx_plot0, yx_plot)):
            if y == y0:
                continue
            yx_collisions = sorted((yy, xx) for yy, xx in yx_plot
                                   if y0 - txt_height <= yy < y + txt_height
                                   and abs(xx - x) < txt_width * 2
                                   and (yy, xx) != (y, x))
            if not yx_collisions:
                yx_plot[i][0] = y0
                changed = True
                n_changes += 1
                print('@@1', n, i)
            elif yx_collisions[0][0] > y0 + txt_height * 2:
                yx_plot[i][0] = y0
                changed = True
                n_changes += 1
                print('@@2', n, i)
            else:
                yx_collisions2 = [(y0, x0)] + yx_collisions + [(y, x)]
                for j, (dy, dx) in enumerate(np.diff(yx_collisions2, axis=0)):
                    if dy > txt_height * 2: # Room for text?
                        yx_plot[i][0] = yx_collisions2[j][0] + txt_height
                        changed = True
                        n_changes += 1
                        print('@@3', n, i)
                        break
            if not changed and y > yx_collisions[-1][0] + 2 * txt_height:
                yx_plot[i][0] = yx_collisions[-1][0] + 2 * txt_height
                changed = True
                n_changes += 1
                print('@@4', n, i)

        print('**** n=%d,n_changes=%d,changed=%s' % (n, n_changes, changed))
        for i, ((y0, x0), (y, x), (y1, x1)) in enumerate(zip(yx_plot0, yx_plot, yx_plot1)):
            assert y0 <= y <= y1, (y0, y, y1)
        if not changed:
            break
        # if all(y == y1 for (y, x), (y1, x1) in zip(yx_plot, yx_plot1)):
        #     break

    # assert n == 1  # !@#$
    return [(x, y) for y, x in yx_plot]


def get_peak_coords(history):
    tsm, peak_idx = history
    assert tsm.index.max() <= MAX_DATE
    xy_data = [(tsm.index[idx], tsm.iloc[idx]) for idx in peak_idx[:n_peaks]]

    def key_plot(x, y):
        return -y, x

    xy_data.sort(key=lambda k: key_plot(*k))
    return xy_data


def plot_loc_date(ax, label, history, n_peaks=N_PEAKS):
    """Plot LoC vs date for time series in history
    """
    # TODO Show areas !@#$
    tsm, peak_ixy = history
    assert tsm.index.max() <= MAX_DATE

    # print('plot_loc_date: %s' % ts_summary(tsm))

    tsm.plot(label=label, ax=ax)

    X0, X1, Y0, Y1 = ax.axis()
    print('***plot_loc_date:', ax.axis(), X1 - X0, len(peak_ixy), n_peaks)
    DX = (X1 - X0) / 50 #  5
    DY = (Y1 - Y0) / 15 # 10

    if not peak_ixy:
        return
    peak_ixy = sorted(peak_ixy)

    x0 = tsm.index[0]
    # !@#$ TODO Get actual text size
    txt_height = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.15 * (plt.xlim()[1] - plt.xlim()[0])

    # NOTE: The following code assumes xy_data is sorted by y

    xy_plot = [(delta_days(x0, x) + X0, y ) for _, x, y in peak_ixy]

    print('txt_width=%.1f,txt_height=%.1f' % (txt_width, txt_height))

    xy_text = _get_xy_text(xy_plot, txt_width, txt_height)
    plt.ylim(plt.ylim()[0], max(plt.ylim()[1], max(y for _, y in xy_text) + 2 * txt_height))
    X0, X1, Y0, Y1 = ax.axis()
    print('**!plot_loc_date:', ax.axis(), X1 - X0)

    # Write high labels first so that lower label text overwrites higher label arrows
    def key_text((i, x, y), (x_p, y_p), (x_t, y_t)):
        return -y_t, x_t, i

    i_data_plot_text = zip(peak_ixy, xy_plot, xy_text)
    i_data_plot_text.sort(key=lambda k: key_text(*k))

    # Label the peaks
    for (i, x, y), (x_p, y_p), (x_t, y_t) in i_data_plot_text:
        ax.annotate('%d) %s, %.0f' % (i + 1, date_str(x), y),
                    xy=(x_p, y_p), xytext=(x_t, y_t),
                    arrowprops={'facecolor': 'red'},
                    horizontalalignment='center', verticalalignment='bottom')
        assert X0 <= x_p <= X1, (X0, x_p, X1)
        assert X0 <= x_t <= X1, (X0, x_t, X1)
        assert Y0 <= y_t <= Y1, (Y0, y_t, Y1)


def plot_show(ax, blame_map, report_map, do_show, graph_path, do_legend):
    """Show and/or save the current markings in axes `ax`
    """

    X0, X1, Y0, Y1 = ax.axis()
    print('*!*plot_show:', ax.axis(), X1 - X0)

    repo_summary = blame_map._repo_map.summary
    rev_summary = blame_map._rev_map.summary
    path_list = report_map.path_list

    path_str = ''
    if len(path_list) == 1:
        path_str = '/%s' % path_list[0]
    elif len(path_list) > 1:
        path_str = '/[%s]' % '|'.join(path_list)

    if do_legend:
        plt.legend(loc='best')
    ax.set_title('%s%s code age (as of %s)\n'
                 'commit=%s : "%s"' % (
                 repo_summary['remote_name'],
                 path_str,
                 rev_summary['date'],
                 truncate_hash(rev_summary['commit']),
                 blame_map.get_description(),  # !@#$ name | description
                 ))
    ax.set_xlabel('date')
    ax.set_ylabel('LoC / day')
    if graph_path:
        plt.savefig(graph_path)
    if do_show:
        plt.show()
        assert False
    print('shown')


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
        self.summary = summary
        self.catalog = {k: kwargs.get(k, v()) for k, v in self.__class__.TEMPLATE.items()}

    def load(self):
        catalog = load_object(self.make_path('data.pkl'), {})
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
            for k, v in self.catalog.items():
                assert k in catalog, (k, catalog.keys(), path)
                catalog[k].update(v)
            self.catalog = catalog

        mkdir(self.base_dir)

        save_object(path, self.catalog)
        print('saved %s' % path)

        path = self.make_path('summary')
        open(os.path.join(path), 'wt').write(repr(self.summary))
        assert os.path.exists(path), os.path.abspath(path)

        manifest = {k: len(v) for k, v in self.catalog.items()}
        path = self.make_path('manfiest')
        open(os.path.join(path), 'wt').write(repr(manifest))
        assert os.path.exists(path), os.path.abspath(path)

    def __repr__(self):
        return repr({k: len(v) for k, v in self.catalog.items()})


class BlameRepoMap(Persistable):
    """Repository level persisted data structures
        Currently this is hash_date_author.
    """
    TEMPLATE = {'hash_date_author': lambda: {}}


class BlameRevMap(Persistable):
    """Revision level persisted data structures
        The main stucture is path_hash_loc.
    """
    # !@#$ don;t need extensions
    # pickle and zip catalog !@#$
    DICT = lambda: {}
    COUNT = lambda: Counter()
    SET = lambda: set()

    TEMPLATE = {
        'path_hash_loc': DICT,
        'path_set': SET,
        'author_set': SET,
        'ext_set': SET,

        # optional
        'exts_good': COUNT,
        'exts_good_loc': COUNT,
        'exts_bad': COUNT,
        'exts_ignored': COUNT,
    }


class BlameMap(object):
    """A BlameMap contains data from git blame that are used to compute reports
        This data can take a long time to generate so we allow it to be saved to and loaded from
        disk so that it can be reused between runs
    """

    # git.stats directory layout
    # root\             ~/git.stats/
    #   repo\           ~/git.stats/papercut
    #     rev\          ~/git.stats/papercut/2015-11-16.3f4632c6
    #       reports\    ~/git.stats/papercut/2015-11-16.3f4632c6/reports/tools_page-analysis-tools
    root_dir = os.path.join(os.path.expanduser('~'), 'git.stats')

    def __init__(self, repo_summary, rev_summary):
        self.repo_summary = repo_summary
        self.rev_summary = rev_summary
        repo_dir = os.path.join(self.__class__.root_dir, repo_summary['remote_name'])
        self._repo_map = BlameRepoMap(repo_summary, repo_dir)
        rev_dir = os.path.join(self._repo_map.base_dir, truncate_hash(rev_summary['commit']))
        self._rev_map = BlameRevMap(rev_summary, rev_dir)

    def load(self):
        self._repo_map.load()
        self._rev_map.load()

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

    def get_description(self):
        summary = self._rev_map.summary
        return summary.get('branch', summary['description'])

    def get_peer_blames(self, file_list):
        for branch in blame_map.brach_list:
            diff_list = git_diff(current_branch, branch)
            file_list = file_list - diff_list
            common_blame = file_list & branch.path_set
            for path in common_blame:
                path_hash_loc[path] = branch.path_hash_loc[path]
                path_set.add(path)

    def update(self, file_list):
        """Compute base statistics over whole repository:hash
            blame all files in `path_list`
            Update: hash_date_author, path_hash_loc for files that are not already in path_hash_loc

            Also update exts_good, exts_good_loc, exts_bad, exts_ignored
        """
        rev_summary = self.rev_summary
        hash_date_author = self.hash_date_author
        path_set = self._rev_map.catalog['path_set']
        author_set = self._rev_map.catalog['author_set']
        ext_set = self._rev_map.catalog['ext_set']

        path_hash_loc = self._rev_map.catalog['path_hash_loc']
        exts_good = self._rev_map.catalog['exts_good']
        exts_good_loc = self._rev_map.catalog['exts_good_loc']
        exts_bad = self._rev_map.catalog['exts_bad']
        exts_ignored = self._rev_map.catalog['exts_ignored']

        assert isinstance(hash_date_author, dict), type(hash_date_author)

        n_all_files = len(file_list)
        file_list = [path for path in file_list if path not in path_set]
        n_files = len(file_list)
        print('%d files (%d total)' % (n_files, n_all_files))

        paths0 = len(path_set)
        loc0 = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
        commits0 = len(hash_date_author)
        start = time.time()
        blamed = 0
        last_loc = loc0

        for i, path in enumerate(file_list):
            path_set.add(path)
            if i % 100 == 0:
                duration = time.time() - start

                loc = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
                if loc != last_loc:
                    rate = blamed / duration if duration >= 1.0 else 0
                    print('i=%d of %d(%.1f%%),files=%d,loc=%d,commits=%d,dt=%.1f,r=%.1f,path=%s' % (
                          i, n_files, 100 * i / n_files, blamed,
                          loc - loc0,
                          len(hash_date_author) - commits0,
                          duration, rate,
                          printable(path)))
                    sys.stdout.flush()
                    last_loc = loc
            ext = get_ext(path)

            if ext in IGNORED_EXTS:
                exts_ignored[ext] += 1
                continue
            if any(pat in path for pat in IGNORED_PATTERNS):
                continue

            try:
                max_date=rev_summary['date']
                text = git_blame_text(path)
                get_text_hash_loc(hash_date_author, path_hash_loc, path_set, author_set, ext_set,
                                  text, path, max_date)
            except:
                if not os.path.exists(path):
                    print('%s no longer exists' % path, file=sys.stderr)
                    exts_bad[ext] += 1
                    continue
                raise

            exts_good[ext] += 1
            # exts_good_loc[ext] += sum(hash_loc.values())
            blamed += 1
            # !@#$ Move all these minor data updates to get_text_hash_loc

        print('~' * 80)
        duration = time.time() - start
        loc = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
        rate = blamed / duration if duration >= 1.0 else 0
        print('%d files,%d blamed,%d lines,%d commits,dt=%.1f,rate=%.1f' % (len(file_list), blamed,
              loc, len(hash_date_author), duration, rate))

        return len(path_set) > paths0


def none_len(n, o):
    """Debugging code for checkking args
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
    print('_filter_list', len(s_list), pattern, end=' : ')
    if pattern is None:
        print(None)
        return None
    regex = re.compile(pattern, re.IGNORECASE)
    res = {s for s in s_list if regex.search(s)}
    print(len(res), sorted(res)[:10])
    return res


def filter_path_hash_loc(blame_map, path_hash_loc, file_list=None, author_list=None, ext_list=None):
    """Trim  `path_hash_loc` down to files in `file_list` authors in `author_list` and extensions
        in `ext_list'
        Note: Does NOT modify path_hash_loc
        # TODO: inplace?
        !@#$ does hash_date_author need to be filtered?

    """

    hash_date_author = blame_map.hash_date_author
    path_hash_loc0 = path_hash_loc

    all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

    if file_list is not None:
        file_list = set(file_list)
    if author_list is not None:
        author_list = set(author_list)

    print('filter_path_hash_loc 1: hash_date_author=%d,path_hash_loc=%d=%d' % (
          len(hash_date_author),
          len(path_hash_loc), sum(sum(v.values()) for v in path_hash_loc.values()) ))
    print('file_list,author_list,ext_list=%s' %
          [none_len(n, o) for n, o in ('file_list', file_list),
                                   ('author_list', author_list),
                                   ('ext_list', ext_list)])
    if author_list is not None:
        print('author_list: %d %s' % (len(author_list), sorted(author_list)[:10]))

    if ext_list is not None:
        assert isinstance(ext_list, set), type(ext_list)

    if file_list or ext_list:

        path_hash_loc = {path: hash_loc for path, hash_loc in path_hash_loc.items()
                         if (not file_list or path in file_list) and
                            (not ext_list or get_ext(path) in ext_list)
                         }

    print('filter_path_hash_loc 2: hash_date_author=%d,path_hash_loc=%d=%d' % (
          len(hash_date_author),
          len(path_hash_loc), sum(sum(v.values()) for v in path_hash_loc.values()) ))

    if author_list:
        author_list = set(author_list)
        hash_set = {hsh for hsh, (_, author) in hash_date_author.items() if author in author_list}
        path_hash_loc = {path: {hsh: loc for hsh, loc in hash_loc.items() if hsh in hash_set}
                         for path, hash_loc in path_hash_loc.items() }

    print('filter_path_hash_loc 3: hash_date_author=%d,path_hash_loc=%d=%d' % (
          len(hash_date_author),
          len(path_hash_loc), sum(sum(v.values()) for v in path_hash_loc.values()) ))
    current_authors = {a for _, a in hash_date_author.values()}

    def xxx(lst):
        if lst is None: return None
        return len(lst), sorted(lst)

    print('author_list    :', xxx(author_list))
    print('current_authors:', xxx(current_authors))

    all_authors = {a for _, a in blame_map.hash_date_author.values()}
    assert all_authors == all_authors0, (all_authors, all_authors0)

    return path_hash_loc


class ReportMap(object):
    """ReportMaps contain data for reports. Unlike BlameMaps they don't make git calls, the only
        filter day, write reports and plot graphs so they _should_ be fast
    """

    def __init__(self, blame_map, path_hash_loc, path_list, reports_dir, file_list, author_pattern,
        ext_pattern, author_list=None):
        # assert isinstance(catalog, dict), type(catalog)
        assert author_pattern is None or isinstance(author_pattern, str), author_pattern
        assert ext_pattern is None or isinstance(ext_pattern, str), ext_pattern
        assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list

        # !@#$ Remove all the all_authors asserts
        all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

        authors = {author for _, author in blame_map.hash_date_author.values()}
        exts = {get_ext(path) for path in path_hash_loc.keys()}
        self.author_list = _filter_list(authors, author_pattern)
        if not self.author_list:
            self.author_list = author_list
        elif author_list is not None:
            self.author_list &= set(author_list)

        self.ext_list = _filter_list(exts, ext_pattern)

        path_hash_loc = filter_path_hash_loc(blame_map, path_hash_loc, file_list, self.author_list, self.ext_list)

        print('self.author_list:', self.author_list)
        print('self.ext_list:', self.ext_list)
        # print('self.blame_map:', self.blame_map)

        assert ':' not in reports_dir[2:], reports_dir
        self.reports_dir = reports_dir
        self.path_list = path_list

        all_authors = {a for _, a in blame_map.hash_date_author.values()}
        assert all_authors == all_authors0, (all_authors, all_authors0)
        self.path_hash_loc = path_hash_loc


def save_tables(blame_map, report_map):
    print('save_tables')
    reports_dir = report_map.reports_dir
    hash_date_author = blame_map.hash_date_author
    path_hash_loc = blame_map.path_hash_loc
    exts_good = blame_map._rev_map.catalog['exts_good']
    exts_good_loc = blame_map._rev_map.catalog['exts_good_loc']
    exts_bad = blame_map._rev_map.catalog['exts_bad']
    exts_ignored = blame_map._rev_map.catalog['exts_ignored']

    if not path_hash_loc:
        print('No files to process')
        return False

    if False:
        df_ext_author_files, df_ext_author_loc = derive_blame(path_hash_loc, hash_date_author, exts_good)

        df_guild_author_files = ext_to_guild(df_ext_author_files)
        df_guild_author_loc = ext_to_guild(df_ext_author_loc)
        # print_counts('exts_good', exts_good)

        print('save_tables: hash_date_author=%d,path_hash_loc=%d' % (len(hash_date_author), len(path_hash_loc)))

        exts = [(k, v, exts_good_loc.get(k, -1)) for k, v in exts_good.items()]
        b_exts = exts_bad.items()
        i_exts = exts_ignored.items()

        df_exts = DataFrame(sorted(exts, key=lambda kv: (-kv[1], -kv[2], kv[0])),
                            columns=['Extension', 'Count', 'LoC'])
        df_bad_exts = DataFrame(sorted(b_exts, key=lambda kv: (-kv[1], kv[0])),
                                columns=['Extension', 'Count'])
        df_ignored_exts = DataFrame(sorted(i_exts, key=lambda kv: (-kv[1], kv[0])),
                                    columns=['Extension', 'Count'])

        def make_path(key):
            return os.path.join(reports_dir, '%s.csv' % key)

        mkdir(reports_dir)
        df_exts.to_csv(make_path('exts_good'), index=False)
        df_bad_exts.to_csv(make_path('exts_bad'), index=False)
        df_ignored_exts.to_csv(make_path('exts_ignored'), index=False)
        df_append_totals(df_ext_author_files).to_csv(make_path('ext_author_files'))
        df_append_totals(df_ext_author_loc).to_csv(make_path('ext_author_loc'))
        df_append_totals(df_guild_author_files).to_csv(make_path('guild_author_files'))
        df_append_totals(df_guild_author_loc).to_csv(make_path('guild_author_loc'))

    return True


DATE_INF_NEG = Timestamp('1911-11-22 11:11:11 -0700')
DATE_INF_POS = Timestamp('2111-11-22 11:11:11 -0700')


def get_top_authors(blame_map, report_map):
    """Get top authors in `report_map`
    """

    hash_date_author = blame_map.hash_date_author
    path_hash_loc = report_map.path_hash_loc

    print('get_top_authors: 2')
    # hash_loc = {hsh: loc} over all hashes
    author_loc_dates = defaultdict(lambda: [0, DATE_INF_POS, DATE_INF_NEG])
    i = 0
    t0 = time.time()
    for h_l in path_hash_loc.values():
        for hsh, loc in h_l.items():
            date, author = hash_date_author[hsh]
            # loc_dates = author_loc_dates.get(author, [0, DATE_INF_POS, DATE_INF_NEG])
            loc_dates = author_loc_dates[author]
            loc_dates[0] += loc
            loc_dates[1] = min(loc_dates[1], date)
            loc_dates[2] = max(loc_dates[2], date)

            if i % 20000 == 0:
                dt = time.time() - t0
                if dt:
                    print('%8d, %5.1f sec, %.1f items/sec' % (
                          i, dt, i /dt))
            i += 1

    assert author_loc_dates
    print('get_top_authors: 2')
    # pprint(author_loc_dates)

    return author_loc_dates, sorted(author_loc_dates.keys(), key=lambda a: -author_loc_dates[a][0])


def analyze_blame(blame_map, report_map):
    """TODO: Add filter by extensions and authors
    !@#$% => analyze a ReportMap
    """
    hash_date_author = blame_map.hash_date_author
    path_hash_loc = report_map.path_hash_loc

    print('analyze_blame: 1')

    # dates = dates of all commits
    dates = [date for date, _ in hash_date_author.values()]

    # hash_loc = {hsh: loc} over all hashes
    hash_loc = Counter()
    for path, h_l in path_hash_loc.items():
        for hsh, loc in h_l.items():
            hash_loc[hsh] += loc

    print('analyze_blame: 1a')

    # hsh_by_loc = sorting key for hash_loc  rename to key_loc
    hsh_by_loc = sorted(hash_loc.keys(), key=lambda k: hash_loc[k])
    total = sum(hash_loc.values())

    # Populate the following dicts
    author_loc = Counter()              # {author: loc}
    author_dates = {}                   # {author: (min date, max date)}
    author_date_hash_loc = {}           # {author: [(date, hsh, loc)]}

    print('analyze_blame: 1b: hash_loc=%d,hash_date_author=%d' % (
          len(hash_loc), len(hash_date_author)))

    assert hash_loc and hash_date_author

    # hash_loc can be very big, e.g. 200,000 for linux source
    t0 = time.time()
    for i, (hsh, loc) in enumerate(hash_loc.items()):
        date, author = hash_date_author[hsh]
        author_loc[author] += loc
        if author not in author_dates:
            author_dates[author] = [date, date]
        else:
            date_min, date_max = author_dates[author]
            author_dates[author] = [min(date, date_min), max(date, date_max)]
        # !@#$ makeauthor_date_hash_loc defaultdict(list)
        if author not in author_date_hash_loc.keys():
            author_date_hash_loc[author] = []
        author_date_hash_loc[author].append((date, hsh, loc))
        if i % 20000 == 0:
            dt = time.time() - t0
            if dt:
                print('%8d %.3f, %5.1f sec, %.1f items/sec' % (i, i / len(hash_loc), dt, i /dt))

    assert author_loc
    print('analyze_blame: 2')

    for author in author_date_hash_loc.keys():
        author_date_hash_loc[author].sort()

    if False:
        # Write some reports
        print('=' * 80)
        print('Biggest commits by author')
        for author in sorted(author_loc.keys(), key=lambda k: -author_loc[k]):
            assert len(author) < 50, author
            date_hash_loc = sorted(author_date_hash_loc[author], key=lambda dhl: (-dhl[2], dhl[0]))
            print('-' * 80)
            print('%s: %d commits, %d loc' % (author, len(date_hash_loc),
                  sum(loc for _, _,loc in date_hash_loc)))
            for i, (date, hsh, loc) in enumerate(date_hash_loc[:10]):
                print('%3d: %8s, %s, %5d' % (i, hsh, date, loc))

    # author_stats = {author: (loc, (min date, max date), #days, ratio)}
    author_stats = {}
    for author in sorted(author_loc.keys(), key=lambda k: -author_loc[k]):
        loc = author_loc[author]
        dates = author_dates[author]
        days = (dates[1] - dates[0]).days
        ratio = loc / days if days else 0.0
        author_stats[author] = (loc, dates, days, ratio)
    assert author_stats

    if False:
        print('~' * 80)
        print('Author stats')
        total = sum(author_loc.values())
        average_loc = total / len(author_loc)

        def author_key(key):
            loc, dates, days, ratio = author_stats[key]
            return days < 60, -ratio

        author_list = sorted(author_stats.keys(), key=author_key)
        for author in author_list:
            loc, dates, days, ratio = author_stats[author]
            print('%-20s %6d %4.1f%% %s %4d %4.1f' % (author, loc, loc / total * 100,
                  [date_str(d) for d in dates], days, ratio))
        print('@' * 80)
    print('hash_loc=%d,hash_date_author=%d' % (len(hash_loc), len(hash_date_author)))
    t0 = time.time()

    author_date_loc = defaultdict(list)
    for hsh, loc in hash_loc.items():
        date, author = hash_date_author[hsh]
        author_date_loc[author].append((date, loc))

    print('author_date_loc=%d,%d' % (len(author_date_loc),
                                     sum(len(v) for v in author_date_loc.values())
                                   ))
    print('authors:', sorted(author_date_loc.keys())[:100])
    dt = time.time() - t0
    print('%.2f seconds %.1f items / sec' % (dt, (len(hash_date_author) / dt) if dt else 0))
    print('#' * 60)

    return {'author_list': report_map.author_list,
            'ext_list': report_map.ext_list,
            'hash_date_author': hash_date_author,
            'hash_loc': hash_loc,
            'author_date_loc': author_date_loc,
            'author_date_hash_loc': author_date_hash_loc,
            'author_stats': author_stats}


DAY = pd.Timedelta('1 days') # 24 * 3600 * 1e9

def get_peak_commits(hash_loc, date_hash_loc, history, window=20 * DAY):
    """Return lists of commits around peaks in a time series
    """
    ts, peak_ixy = history
    dt = window / 2

    print('date_hash_loc:', type(date_hash_loc))
    print('date_hash_loc:', len(date_hash_loc))
    assert len(date_hash_loc[0]) == 3, date_hash_loc[0]

    peak_ends = [(x - dt, x + dt) for _, x, _ in peak_ixy]
    for i in xrange(1, len(peak_ends)):
        m0, m1 = peak_ends[i - 1]
        n0, n1 = peak_ends[i]
        if n0 > n1:
            peak_ends[i - 1][1] = peak_ends[i][0] = (peak_list[0] + peak_list[1]) / 2

    peak_commits = []
    for (i, x, y), (m0, m1) in zip(peak_ixy, peak_ends):
        assert isinstance(x, pd.Timestamp), (type(x), x)
        mode_hashes = [hsh for (date, hsh, loc) in date_hash_loc if m0 <= date < m1]
        mode_hashes.sort(key=lambda hsh: -hash_loc[hsh])
        loc = sum(hash_loc[hsh]for hsh in mode_hashes)
        peak_commits.append((loc, x, mode_hashes))
    loc_total = sum(loc for loc, _, _ in peak_commits)

    return loc_total, zip(peak_ixy, peak_commits)


def plot_analysis(blame_map, report_map, history, author, do_show, graph_path):

    print('plot_analysis: before')

    # update figure number
    fig, ax0 = plt.subplots(nrows=1)

    label = None
    ts, peak_idx = history
    if peak_idx:
        is_annotated = True
    print('Time series: %s: %s' % (label, ts_summary(ts)))
    plot_loc_date(ax0, label, (ts, peak_idx))

    print('plot_analysis: plot')
    plot_show(ax0, blame_map, report_map, do_show, graph_path, False)
    print('plot_analysis: after')


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


def write_legend(f, author_date_loc, hash_loc, history, date_hash_loc,
    hash_date_author, author, N_TOP):

    def put(s):
        # print(s)
        f.write('%s\n' % s)

    loc_auth, peak_ixy_commits = get_peak_commits(hash_loc, date_hash_loc, history)
    peak_ixy_commits.sort(key=lambda k: key_ixy_x(*k[0]))
    put('=' * 80)
    put('%s: %d peaks %d LoC' % (author, len(peak_ixy_commits), loc_auth))
    print('write_legend %s' % [author, len(peak_ixy_commits), loc_auth], end=',')

    for (i, x, y), (loc, peak, mode_hashes) in peak_ixy_commits:
        put('.' * 80)
        put('%3d) %d commits %d LoC around %s' % (i + 1, len(mode_hashes), loc, date_str(peak)))
        print('%3d) %d commits %d LoC around %s' % (i + 1, len(mode_hashes), loc, date_str(peak)))

        # !@#$ Replace N_TOP
        for hsh in sorted(mode_hashes[:N_TOP], key=lambda k: hash_date_author[k][0]):
            put('%5d LoC, %s %s' % (hash_loc[hsh], date_str(hash_date_author[hsh][0]),
                git_show_oneline(hsh)))
    # exit()


def save_analysis(blame_map, report_map, analysis, _a_list, do_save, do_show, N_TOP=3):
    """Create a graph (time series + markers)
        a list of commits in for each peak
        + n biggest commits
        <name>.png
        <name>.txt
    """
    # !@#$ graph and legend numbering is different. extract numbering into a separate method
    reports_dir = report_map.reports_dir

    # print('save_analysis: %d, %s, %s' % (len(history), do_save, do_show))
    mkdir(reports_dir)
    assert do_save
    # assert not do_show
    hash_date_author = analysis['hash_date_author']
    hash_loc = analysis['hash_loc']
    _author_date_hash_loc = analysis['author_date_hash_loc']
    author_date_loc = analysis['author_date_loc']
    author_stats = analysis['author_stats']
    ext_list = analysis['ext_list']

    if not _a_list:
        author = '[all]'
    elif len(_a_list) == 1:
        author = _a_list[0]
    else:
        assert False, _a_list
    date_hash_loc = aggregate_author_date_hash_loc(_author_date_hash_loc, _a_list)
    history = make_history(author_date_loc, _a_list)

    # !@#$ history are needed for 'all' and all major authors
    # !@#$ need a better name than history
    report_name = '[%s]' % (concat(sorted(ext_list)) if ext_list else 'all')
    if do_save:
        graph_name = 'history%s.png' % report_name
        legend_name = 'history%s.txt' % report_name
        graph_path = os.path.join(reports_dir, graph_name)
        legend_path = os.path.join(reports_dir, legend_name)
        print('graph_path="%s"' % graph_path)
        print('legend_path="%s"' % legend_path)
        with open(legend_path, 'wt') as f:
            write_legend(f, author_date_loc, hash_loc, history, date_hash_loc,
                         hash_date_author, author, N_TOP)
    else:
        graph_path = None

    # author = 'all' if not a_list else a_list[0]
    plot_analysis(blame_map, report_map, history, author, do_show, graph_path)


def main(path_list, force, author_pattern, ext_pattern, do_save, do_show):

    # assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list

    remote_url, remote_name = git_remote()
    description = git_describe()
    commit = git_current_commit()
    date = git_date(commit)
    date_ymd = date.strftime('%Y-%m-%d')

    # git.stats directory layout
    # root\             ~/git.stats/
    #   repo\           ~/git.stats/papercut
    #     rev\          ~/git.stats/papercut/2015-11-16.3f4632c6
    #       reports\    ~/git.stats/papercut/2015-11-16.3f4632c6/reports/tools_page-analysis-tools
    root_dir = os.path.join(os.path.expanduser('~'), 'git.stats')
    repo_dir = os.path.join(root_dir, remote_name)
    rev_dir = os.path.join(repo_dir, '.'.join([date_ymd, truncate_hash(commit)]))
    # data_dir = os.path.join(rev_dir, 'data')

    path_list = [normalize_path(path) for path in path_list]
    if not path_list or (len(path_list) == 1 and path_list[0] == '.'):
        reports_name = '[root]'
    else:
        reports_name = '.'.join(clean_path(path) for path in path_list)
    reports_dir = os.path.join(rev_dir, 'reports', reports_name)
    for s in path_list:
        print([s, clean_path(s)])
    manifest_path = os.path.join(rev_dir, 'manifest')

    print('remote: %s (%s) %s' % (remote_name, remote_url, date_ymd))

    assert date < MAX_DATE, date

    # !@#$ TODO Add a branches file in rev_dir
    repo_summary = {
        'remote_url': remote_url,
        'remote_name': remote_name,
    }
    rev_summary = {
        'commit': commit,
        'branch': git_current_branch(),
        'description': description,
        'name': git_name(),
        'date': date,
    }
    pprint(repo_summary)
    pprint(rev_summary)

    file_list = git_file_list(path_list)

    blame_map = BlameMap(repo_summary, rev_summary)
    if not force:
        blame_map.load()
    changed = blame_map.update(file_list)
    if changed:
        blame_map.save(force)

    report_map_all = ReportMap(blame_map, blame_map.path_hash_loc, path_list, reports_dir, file_list,
                               author_pattern, ext_pattern)
    author_loc_dates, top_authors = get_top_authors(blame_map, report_map_all)

    top_a_list = [None] + [[a] for a in top_authors]
    reports_dir_list = []
    all_authors0 = {a for _, a in blame_map.hash_date_author.values()}

    for a_list in top_a_list:
        all_authors = {a for _, a in blame_map.hash_date_author.values()}
        assert all_authors == all_authors0, (all_authors, all_authors0)
        reports_dir_author = '[all]' if a_list is None else a_list[0]
        reports_dir_author = os.path.join(reports_dir, clean_path(reports_dir_author))
        if a_list:
            assert a_list[0] in all_authors, (a_list[0], all_authors)
        report_map = ReportMap(blame_map, report_map_all.path_hash_loc,  path_list, reports_dir_author,
                               None, None, None, a_list)
        all_authors = {a for _, a in blame_map.hash_date_author.values()}
        assert all_authors == all_authors0, (all_authors, all_authors0)


        analysis = analyze_blame(blame_map, report_map)

        save_tables(blame_map, report_map)
        save_analysis(blame_map, report_map, analysis, a_list, do_save, do_show)
        print('reports_dir=%s' % os.path.abspath(report_map.reports_dir))
        reports_dir_list.append(os.path.abspath(report_map.reports_dir))

    print('+' * 80)
    print('rev_dir=%s' % os.path.abspath(rev_dir))
    for reports_dir_author in reports_dir_list:
        print('reports_dir=%s' % reports_dir_author)


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

    do_save = True

    options, args = parser.parse_args()

    main(args, options.force, options.author_pattern, options.ext_pattern, do_save, options.do_show)
    print('DONE')
    print('D0NE')
    show_versions()
