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

    TODO:
        hours of day
        days of week
        top times
        group code by functoonality
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

# Set graphing style
matplotlib.style.use('ggplot')
plt.rcParams['axes.color_cycle'] = ['b', 'y', 'k', '#707040', '#404070'] + plt.rcParams['axes.color_cycle'][1:]

print(plt.rcParams['savefig.dpi'])
plt.rcParams['savefig.dpi'] = 300
print(plt.rcParams['savefig.dpi'])
# assert False

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

TIMEZONE = 'Australia/Melbourne'

HASH_LEN = 8

N_PEAKS = 10

def truncate_hash(hsh):
    return hsh[:HASH_LEN]


def date_str(date):
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
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    assert os.path.exists(path), path


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
    data = np.empty(n, dtype=float)
    for i in xrange(n):
        i0 = max(0, i - d0)
        i1 = min(n, i + d1)
        c0 = i0 - (i - d0)
        c1 = (i + d1) - n
        assert len(series[i0:i1]) == len(weights[c0:window - c1])
        data[i] = np.average(series[i0:i1], weights=weights[c0:window - c1])
        assert series[i0:i1].min() <= data[i] <= series[i0:i1].max(), (
               [window, len(series)], i, [i0, i1], [c0, c1, window - c1],
               [series[i0:i1].min(), data[i], series[i0:i1].max()],
               series[i0:i1],
               weights[c0:window - c1])

    return Series(data, index=series.index)


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


def git_description(obj):
    """https://git-scm.com/docs/git-show
    """
    return exec_headline(['git', 'show', '--oneline', '--quiet', obj])


def git_date(obj):
    date_s = exec_headline(['git', 'show', '--pretty=format:%ai', '--quiet', obj])
    return to_timestamp(date_s)


RE_REMOTE_URL =  re.compile(r'(https?://.*/[^/]+(?:\.git)?)\s+\(fetch\)')
RE_REMOTE_NAME = re.compile(r'https?://.*/(.+?)(\.git)?$')


def git_remote():
    """
    $ git remote -v
    origin  https://github.com/FFTW/fftw3.git (fetch)
    origin  https://github.com/FFTW/fftw3.git (push)
    """
    lines = exec_output_lines(['git', 'remote', '-v'], True)
    for ln in lines:
        m = RE_REMOTE_URL.search(ln)
        if not m:
            continue
        remote_url = m.group(1)
        remote_name = RE_REMOTE_NAME.search(remote_url).group(1)
        # print(RE_REMOTE_NAME.search(remote_url).groups())
        # print(remote_url)
        # print(remote_name)
        # exit()
        return remote_url, remote_name
    raise RuntimeError('No remote')


def git_current_branch():
    """git rev-parse --abbrev-ref HEAD
    """
    branch = exec_headline(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == 'HEAD': # Detached HEAD?
        branch = ''  # !@#$ [None]
    return branch


def git_current_commit():
    return exec_headline(['git', 'rev-parse', 'HEAD'])

RE_PATH = re.compile('''[^a-z^0-9^!@#$\-+=_]''', re.IGNORECASE)
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

if False:
    print('-' * 80)
    for i, path in enumerate(['a/b\\c:de\-f1@', '!@#$%^&*()-_+=[]{}']):
        clean = clean_path(path)
        print('%d:"%s"=>"%s"' % (i, path, clean))
    exit()


def git_blame_text(path):
    """Return git blame text for file `path`
    """
    # https://coderwall.com/p/x8xbnq/git-don-t-blame-people-for-changing-whitespaces-or-moving-code
    # -l for long hashes
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

MAX_DATE = Timestamp('2015-11-22 14:11:54 -0700')

def get_text_hash_loc(hash_date_author, text, path, max_date=None):

    if max_date is None:
        max_date = MAX_DATE

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

        # Linux code
        # drivers/gpu/drm/i915/intel_overlay.c:381
        # 12ca45fea91cfbb09df828bea958b47348caee6d drivers/gpu/drm/i915/intel_overlay.c (Daniel Vetter               2037-04-25 10:08:26 +0200  381) }
        if date > max_date:
            print('Bogus timestamp: %s:%d\n%s' % (path, i + 1, ln[:200]))
            continue

        hash_loc[hsh] += 1
        hash_date_author[hsh] = (date, author)
    return hash_loc


def get_file_hash_loc(hash_date_author, path, max_date=None):
    text = git_blame_text(path)
    return get_text_hash_loc(hash_date_author, text, path, max_date)


def derive_blame(path_hash_loc, hash_date_author, exts_good):
    """Compute summary tables over whole repository:hash
      !@#$ Either filter this to report or compute while blaming
      or limit to top authors <===
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
def _analyze_time_series(loc, date, window=60):
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


def make_history(author_date_loc, author_list):
    """Return a history for all authors in `author_list`
    """
    # date_loc could be a Series !@#$
    # for author, date_loc in author_date_loc.items():
    #     date, loc = zip(*date_loc)
    #     assert delta_days(min(date), max(date)) > 30, (author, min(date), max(date))

    date, loc = [], []
    assert author_list
    for author in author_list:
        assert author_date_loc[author]
        d, l = zip(*author_date_loc[author])

        date.extend(d)
        loc.extend(l)
    # assert delta_days(min(date), max(date)) > 30, (author, min(date), max(date))
    ts, peak_idx = _analyze_time_series(loc, date)
    # print('make_history: ts=%s' % ts_summary(ts, author_list))
    return ts, peak_idx


def _get_y_text(xy_plot, txt_width, txt_height):
    """Return y positions of text labels for points `xy_plot`
        1) Offset text upwards by txt_width
        2) Remove collisions
    """
    yx_plot = [[y + txt_height, x] for x, y in xy_plot]
    y_text = [y for y, _ in yx_plot]
    for i, (y, x) in enumerate(yx_plot):
        yx_collisions = sorted((yy, xx) for yy, xx in yx_plot if yy > y - txt_height
                               and abs(xx - x) < txt_width * 2 and (yy, xx) != (y, x))
        if not yx_collisions or abs(yx_collisions[0][0] - y) >= txt_height:
            continue

        yx_plot[i][0] = y_text[i] = yx_collisions[-1][0] + txt_height
        for k, (dy, dx) in enumerate(np.diff(yx_collisions, axis=0)):
            if dy > txt_height * 2: # Room for text>
                yx_plot[i][0] = y_text[i] = yx_collisions[k][0] + txt_height

    return y_text


def plot_loc_date(ax, label, history, n_peaks=N_PEAKS):
    """Plot LoC vs date for time series in history
    """
    # TODO Show areas !@#$
    tsm, peak_idx = history

    # print('plot_loc_date: %s' % ts_summary(tsm))

    tsm.plot(label=label, ax=ax)

    X0, X1, Y0, Y1 = ax.axis()
    print('***plot_loc_date:', ax.axis(), X1 - X0, len(peak_idx), n_peaks)
    DX = (X1 - X0) / 50 #  5
    DY = (Y1 - Y0) / 15 # 10

    if not peak_idx:
        return

    x0 = tsm.index[0]
    txt_height = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.15 * (plt.xlim()[1] - plt.xlim()[0])

    # NOTE: The following code assumes xy_data is sorted by y
    xy_data = [(tsm.index[idx], tsm.iloc[idx]) for idx in peak_idx[:n_peaks]]
    xy_data.sort(key=lambda xy: xy[::-1])
    xy_plot = [(delta_days(x0, x) + X0, y ) for x, y in xy_data]
    # xy_plot2 = [(x, y + txt_height) for x, y in xy_plot]

    print('txt_width=%.1f,txt_height=%.1f' % (txt_width, txt_height))

    text_pos_y = _get_y_text(xy_plot, txt_width, txt_height)
    plt.ylim(plt.ylim()[0], max(plt.ylim()[1], max(text_pos_y) + 2 * txt_height))
    X0, X1, Y0, Y1 = ax.axis()
    print('**!plot_loc_date:', ax.axis(), X1 - X0,)

    for i in range(len(text_pos_y)):
        yy = text_pos_y[i]
        x, y = xy_plot[i]
        print('$ %2d: [%.1f, %.1f] %.2f' % (i, x, y, yy - y))

    # Mark the peaks

    for i, (x, y) in sorted(enumerate(xy_data), key=lambda ixy: tuple(ixy[::-1])):

        x_d, y_d = xy_plot[i]
        x_t, y_t = x_d, text_pos_y[i]
        ax.annotate('%d) %s, %.0f' % (i + 1, date_str(x), y),
                    # xy=(x_d, y_d), xytext=(x_d, y_d + DY),
                    xy=(x_d, y_d), xytext=(x_t, y_t),
                    arrowprops={'facecolor': 'red',
                    # 'arrowstyle': 'fancy'
                    },
                    horizontalalignment='center', verticalalignment='bottom')
        # print('###', x, delta_days(x, x0), y)
        assert X0 <= x_d <= X1, (X0, x_d, X1)
        assert X0 <= x_t <= X1, (X0, x_t, X1)
        assert Y0 <= y_t <= Y1, (Y0, y_t, Y1)


def plot_show(ax, report_map, do_show, graph_path, do_legend):
    """Show and/or save the current markings in axes `ax`
    """
    summary = report_map.summary
    path_list = report_map.path_list

    path_str = ''
    if len(path_list) == 1:
        path_str = '/%s' % path_list[0]
    elif len(path_list) > 1:
        path_str = '/[%s]' % '|'.join(path_list)

    if do_legend:
        plt.legend(loc='best')
    ax.set_title('%s%s code age (as of %s)\n'
                 'commit=%s,branch="%s"' % (
                 summary['remote_name'],
                 path_str,
                 date_str(summary['date']),
                 truncate_hash(summary['commit']),
                 summary['branch'],
                 ))
    ax.set_xlabel('date')
    ax.set_ylabel('LoC / day')
    if graph_path:
        plt.savefig(graph_path)
    if do_show:
        plt.show()
        assert False
    print('shown')


class BlameMap(object):
    """A BlameMap contains data from git blame that are used to compute reports
        This data can take a long time to generate so we allow it to be saved to and loaded from
        disk so that it can be reused between runs

    """
    # !@#$ don;t need extensions
    # pickle and zip catalog !@#$
    DICT = (lambda: {}, '.pkl')
    COUNT = (lambda: Counter(), '.pkl')
    SET = (lambda: set(), '.pkl')

    TEMPLATE = {
        'hash_date_author': DICT,
        'path_hash_loc': DICT,
        'path_set': SET,

        # optional
        'exts_good': COUNT,
        'exts_good_loc': COUNT,
        'exts_bad': COUNT,
        'exts_ignored': COUNT,
    }

    def make_path(self, key):
        return os.path.join(self.data_dir, '%s%s' % (key, BlameMap.TEMPLATE[key][1]))

    def __init__(self, git_summary, data_dir, **kwargs):
        self.data_dir = data_dir
        self.git_summary = git_summary
        self.catalog = {k: kwargs.get(k, v[0]()) for k, v in BlameMap.TEMPLATE.items()}
        pprint(self.catalog)

    def load_catalog(self):
        catalog = {}
        for k, (f, _) in BlameMap.TEMPLATE.items():
            catalog[k] = load_object(self.make_path(k), f())

        # temp hack !@#$
        path_set = catalog['path_set']
        path_hash_loc = catalog['path_hash_loc']
        for path in path_hash_loc.keys():
            path_set.add(path)

        return catalog

    def load(self):
        catalog = self.load_catalog()
        for k, v in catalog.items():
            self.catalog[k].update(v)
        path = os.path.join(self.data_dir, 'summary')
        if os.path.exists(path):
            self.git_summary = eval(open(path, 'rt').read())

    def save(self):
        # Load before saving in case another instance of this script is running
        catalog = self.load_catalog()
        for k, v in self.catalog.items():
            catalog[k].update(v)
        self.catalog = catalog

        for k, v in self.catalog.items():  # !@#$ map is key word. Change
            path = self.make_path(k)
            save_object(path, v)
            print('saved %s: %s' % (k, path))
        open(os.path.join(self.data_dir, 'summary'), 'wt').write(repr(self.git_summary))
        assert os.path.exists(path), os.path.abspath(path)


def update_blame_map(blame_map, file_list):
    """Compute base statistics over whole repository:hash
        blame all files in `path_list`
        Update: hash_date_author, path_hash_loc for files that are not already in path_hash_loc

        Also update exts_good, exts_good_loc, exts_bad, exts_ignored

    """
    summary = blame_map.git_summary
    path_set = blame_map.catalog['path_set']
    hash_date_author = blame_map.catalog['hash_date_author']
    path_hash_loc = blame_map.catalog['path_hash_loc']
    exts_good = blame_map.catalog['exts_good']
    exts_good_loc = blame_map.catalog['exts_good_loc']
    exts_bad = blame_map.catalog['exts_bad']
    exts_ignored = blame_map.catalog['exts_ignored']

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
            h_l = get_file_hash_loc(hash_date_author, path, max_date=summary['date'])
        except:
            if not os.path.exists(path):
                print('%s no longer exists' % path, file=sys.stderr)
                continue
            raise

        if not h_l:
            exts_bad[ext] += 1
            continue

        path_hash_loc[path] = h_l
        exts_good[ext] += 1
        exts_good_loc[ext] += sum(h_l.values())
        blamed += 1

    print('~' * 80)
    duration = time.time() - start
    loc = sum(sum(hash_loc.values()) for hash_loc in path_hash_loc.values())
    rate = blamed / duration if duration >= 1.0 else 0
    print('%d files,%d blamed,%d lines,%d commits,dt=%.1f,rate=%.1f' % (len(file_list), blamed,
          loc, len(hash_date_author), duration, rate))

    return len(path_set) > paths0


def none_len(n, o):
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


def filter_blame(catalog, file_list=None, author_list=None, ext_list=None):
    """
        Trim `hash_date_author`, `path_hash_loc` down to authors in `author_list` and extensions
        in `ext_list'
        Note: Does NOT modify catalog
        # TODO: inplace?
        !@#$ does hash_date_author need to be filtered?

    """

    hash_date_author = catalog['hash_date_author']
    path_hash_loc = catalog['path_hash_loc']
    path_hash_loc0 = path_hash_loc

    if file_list is not None:
        file_list = set(file_list)
    if author_list is not None:
        author_list = set(author_list)

    print('filter_blame 1: hash_date_author=%d,path_hash_loc=%d' % (
          len(hash_date_author),
          len(path_hash_loc)))
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

    print('filter_blame 2: hash_date_author=%d,path_hash_loc=%d' % (
          len(hash_date_author),
          len(path_hash_loc)))
    # print(path_hash_loc.items()[0])
    if author_list:
        author_list = set(author_list)
        hash_set = {hsh for hsh, (_, author) in hash_date_author.items() if author in author_list}
        path_hash_loc = {path: {hsh: loc for hsh, loc in hash_loc.items() if hsh in hash_set}
                         for path, hash_loc in path_hash_loc.items() }

    print('filter_blame 3: hash_date_author=%d,path_hash_loc=%d' % (
          len(hash_date_author),
          len(path_hash_loc)))
    # print(path_hash_loc.values()[0])

    if path_hash_loc is path_hash_loc0:
        return catalog # !@#$
        path_hash_loc = copy.deepcopy(path_hash_loc)
        hash_date_author = copy.deepcopy(hash_date_author)
    else:
        hashes = {hsh for hash_loc in path_hash_loc.values() for hsh in hash_loc.keys()}
        hash_date_author = {hsh: date_author for hsh, date_author in hash_date_author.items()
                            if hsh in hashes}

    print('filter_blame 4: hash_date_author=%d,path_hash_loc=%d' % (
          len(hash_date_author),
          len(path_hash_loc)))
    assert hash_date_author and path_hash_loc

    # !@#$ hack. need to filter other catalog members
    catalog = catalog.copy()
    catalog['hash_date_author'] = hash_date_author
    catalog['path_hash_loc'] = path_hash_loc
    return catalog

    return {'hash_date_author': hash_date_author,
            'path_hash_loc': path_hash_loc}


class ReportMap(object):

    def __init__(self, summary, catalog, path_list, reports_dir, file_list, author_pattern,
        ext_pattern, author_list=None):
        assert isinstance(catalog, dict), type(catalog)
        assert author_pattern is None or isinstance(author_pattern, str), author_pattern
        assert ext_pattern is None or isinstance(ext_pattern, str), ext_pattern
        assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list

        self.summary = summary
        hash_date_author = catalog['hash_date_author']
        path_hash_loc = catalog['path_hash_loc']
        authors = {author for _, author in hash_date_author.values()}
        exts = {get_ext(path) for path in path_hash_loc.keys()}
        self.author_list = _filter_list(authors, author_pattern)
        if not self.author_list:
            self.author_list = author_list
        elif author_list is not None:
            self.author_list &= set(author_list)

        self.ext_list = _filter_list(exts, ext_pattern)

        self.catalog = filter_blame(catalog, file_list, self.author_list, self.ext_list)
        print('self.author_list:', self.author_list)
        print('self.ext_list:', self.ext_list)
        print('self.catalog:', len(self.catalog))

        self.reports_dir = reports_dir
        self.path_list = path_list


def save_tables(report_map):
    print('save_tables')
    catalog = report_map.catalog
    reports_dir = report_map.reports_dir
    hash_date_author = catalog['hash_date_author']
    path_hash_loc = catalog['path_hash_loc']
    exts_good = catalog['exts_good']
    exts_good_loc = catalog['exts_good_loc']
    exts_bad = catalog['exts_bad']
    exts_ignored = catalog['exts_ignored']

    if not path_hash_loc:
        print('No files to process')
        return False
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


def get_top_authors(report_map):
    """Get top authors in `report_map`
    """

    catalog = report_map.catalog
    hash_date_author = catalog['hash_date_author']
    path_hash_loc = catalog['path_hash_loc']

     # hash_loc = {hsh: loc} over all hashes
    hash_loc = Counter()
    for  h_l in path_hash_loc.values():
        for hsh, loc in h_l.items():
            hash_loc[hsh] += loc

    print('get_top_authors: 1')
    # Precompute authors in blame_map !@#$
    author_set = {author for _, author in hash_date_author.values()}
    author_loc_dates = {author: [0, DATE_INF_POS, DATE_INF_NEG] for author in author_set}

    print('get_top_authors: 1b: hash_loc=%d' % len(hash_loc))

    # hash_loc can be very big, e.g. 200,000 for linux source
    t0 = time.time()
    for i, (hsh, loc) in enumerate(hash_loc.items()):
        date, author = hash_date_author[hsh]
        loc_dates = author_loc_dates[author]
        loc_dates[0] += loc
        loc_dates[1] = min(loc_dates[1], date)
        loc_dates[2] = max(loc_dates[2], date)

        if i % 20000 == 0:
            dt = time.time() - t0
            print('%8d %.3f, %5.1f sec, %.1f items/sec' % (i, i / len(hash_loc), dt, i /dt))

    assert author_loc_dates
    print('get_top_authors: 2')

    return author_loc_dates, sorted(author_set, key=lambda a: -author_loc_dates[a][0])


def analyze_blame(report_map):
    """TODO: Add filter by extensions and authors
    !@#$% => analyze a ReportMap
    """
    hash_date_author = report_map.catalog['hash_date_author']
    path_hash_loc = report_map.catalog['path_hash_loc']

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


    # shims = [a for a in author_date_loc.keys() if 'shimmin' in a.lower()]
    # print('shims:', shims)
    # for tim in shims:
    #     print(tim)
    #     pprint(list(enumerate(author_date_loc[tim][:10])))

    print('author_date_loc=%d,%d' % (len(author_date_loc),
                                     sum(len(v) for v in author_date_loc.values())
                                   ))
    print('authors:', sorted(author_date_loc.keys())[:100])
    dt = time.time() - t0
    print('%.2f seconds %.1f items / sec' % (dt, len(hash_date_author) / dt))
    print('#' * 60)

    return {'author_list': report_map.author_list,
            'ext_list': report_map.ext_list,
            'hash_date_author': hash_date_author,
            'hash_loc': hash_loc,
            'author_date_loc': author_date_loc,
            'author_date_hash_loc': author_date_hash_loc,
            'author_stats': author_stats}


DAY = pd.Timedelta('1 days') # 24 * 3600 * 1e9

def describe_peaks(hash_loc, date_hash_loc, history, window=20 * DAY):
    """!@#$ Terrible name
        Give list of commits near a peak in a time series
    """
    ts, peak_idx = history
    dt = window / 2

    modes_hashes = []
    mode_list = ts.index[peak_idx]
    mode_limits = [(mode - dt, mode + dt) for mode in mode_list]
    for i in range(1, len(mode_limits)):
        m0, m1 = mode_limits[i - 1]
        n0, n1 = mode_limits[i]
        if n0 > n1:
            mode_limits[i - 1][1] = mode_limits[i][0] = (mode_list[0] + mode_list[1]) / 2

    for mode, (m0, m1) in zip(mode_list, mode_limits):
        assert isinstance(mode, pd.Timestamp), (type(mode), mode)
        mode_hashes = [hsh for (date, hsh, loc) in date_hash_loc if m0 <= date < m1]
        if not mode_hashes:
            continue
        mode_hashes.sort(key=lambda hsh: -hash_loc[hsh])
        modes_hashes.append((sum(hash_loc[hsh]for hsh in mode_hashes), mode, mode_hashes))
    loc_total = sum(loc for loc, _, _ in modes_hashes)
    return loc_total, modes_hashes


def plot_history(report_map,
    hash_loc, hash_date_author, author_date_hash_loc, author_stats, author_history,
    author_list,
    do_show, graph_path,
    n_top_authors=6,
    n_top_commits=5):

    print('plot_history: before')

    # update figure number
    fig, ax0 = plt.subplots(nrows=1)

    # Author modes
    if False:
        print('"' * 50)
        print('plot_history: %d %s' % (len(author_history), sorted(author_history.keys())))
        pprint({author: ts_summary(ts) for author, (ts, _) in author_history.items()})

    is_annotated = False

    for author in author_list[:n_top_authors]:
        _, dates, days, ratio = author_stats[author]
        # label = '%s %.1f LoC/day, %d LoC, %d days (%s) ' % (
        label = '%s %.1f LoC/day, %d days (%s) ' % (
                author, ratio,
                # sum(loc), !@#$ Add this bak
                int(round(days)),
                ' to '.join(str(d.date()) for d in dates))
        ts, peak_idx = author_history[author]
        if peak_idx:
            is_annotated = True
        print('Time series: %s: %s' % (label, ts_summary(ts)))
        plot_loc_date(ax0, label, (ts, peak_idx))

    assert isinstance(hash_loc, dict), type(hash_loc)
    assert isinstance(author_date_hash_loc, dict), type(author_date_hash_loc)
    assert isinstance(author_history, dict), type(author_history)

    author_modes_hashes = {author: describe_peaks(hash_loc, author_date_hash_loc[author],
                                                  author_history[author])
                           for author in author_list}

    print('plot_history: plot')
    plot_show(ax0, report_map, do_show, graph_path, not is_annotated)
    print('plot_history: after')


def _files_put(files, s):
    for f in files:
        f.write('%s\n' % s)


def aggregate_author_date_hash_loc(author_date_hash_loc, author_list):
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


def save_history(report_map, history, do_save, do_show, N_TOP=3):
    """Create a graph (time series + markers)
        a list of commits in for each peak
        + n biggest commits
        <name>.png
        <name>.txt
    """
    catalog = report_map.catalog
    reports_dir = report_map.reports_dir

    # print('save_history: %d, %s, %s' % (len(history), do_save, do_show))
    mkdir(reports_dir)
    assert do_save
    # assert not do_show
    hash_date_author = history['hash_date_author']
    hash_loc = history['hash_loc']
    author_date_hash_loc = history['author_date_hash_loc']
    author_date_loc = history['author_date_loc']
    author_stats = history['author_stats']
    ext_list = history['ext_list']

    author_loc = {author: sum(loc for _, loc in date_loc)
                  for author, date_loc in author_date_loc.items()}
    author_list = sorted(author_loc.keys(), key=lambda k: -author_loc[k])

    # !@#$ history are needed for 'all' and all major authors
    # !@#$ need a better name than history
    report_name = concat(sorted(ext_list)) if ext_list else 'all'
    report_name = '[%s]' % report_name
    if do_save:
        graph_name = 'history%s.png' % report_name
        legend_name = 'history%s.txt' % report_name
        graph_path = os.path.join(reports_dir, graph_name)
        legend_path = os.path.join(reports_dir, legend_name)
        print('graph_path="%s"' % graph_path)
        print('legend_path="%s"' % legend_path)
        legend_f = open(legend_path, 'wt')
        assert legend_f, legend_path
    else:
        graph_path = legend_path = legend_f = None

    files = []
    # if do_show:
    #     files.append(sys.stdout)
    if legend_f:
        files.append(legend_f)

    def put(s):
        _files_put(files, s)

    print('author_list=%d' % len(author_list))
    for a, author in enumerate(author_list):

        # !@#$ make describe_peaks() work for groups of authors, in particular everyone
        history = make_history(author_date_loc, [author])
        loc_auth, modes_hashes = describe_peaks(hash_loc, author_date_hash_loc[author], history)
        put('=' * 80)
        put('%s: %d peaks %d LoC' % (author, len(modes_hashes), loc_auth))
        print('%s' % [a, author, len(modes_hashes), loc_auth], end=',')

        for i, (loc, mode, mode_hashes) in enumerate(modes_hashes):
            put('.' * 80)
            put('%3d: %d commits %d LoC around %s' % (i, len(mode_hashes), loc, date_str(mode)))
            for hsh in sorted(mode_hashes[:N_TOP], key=lambda k: hash_date_author[k][0]):
                put('%5d LoC, %s %s' % (hash_loc[hsh], date_str(hash_date_author[hsh][0]),
                    git_description(hsh)))

    print()
    if legend_f:
        legend_f.close()

    e_author_date_hash_loc = {'Everyone': aggregate_author_date_hash_loc(author_date_hash_loc,
                              author_list)}
    e_author_stats = {'Everyone': aggregate_author_stats(author_stats, author_list)}
    # All. With peaks
    e_author_history = {'Everyone': make_history(author_date_loc, author_list)}

    plot_history(report_map,
                 hash_loc, hash_date_author, e_author_date_hash_loc, e_author_stats,
                 e_author_history, ['Everyone'], do_show, graph_path)


def main(path_list, force, author_pattern, ext_pattern, do_save, do_show):

    # assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list

    remote_url, remote_name = git_remote()
    commit = git_current_commit()
    date = git_date(commit)
    date_ymd = date.strftime('%Y-%m-%d')
    base_dir = os.path.join('git.stats', '.'.join([date_ymd, truncate_hash(commit)]))
    data_dir = os.path.join(base_dir, 'data')

    path_list = [normalize_path(path) for path in path_list]
    if not path_list or (len(path_list) == 1 and path_list[0] == '.'):
        reports_name = '[root]'
    else:
        reports_name = '.'.join(clean_path(path) for path in path_list)
    reports_dir = os.path.join(base_dir, 'reports', reports_name)
    for s in path_list:
        print([s, clean_path(s)])
    manifest_path = os.path.join(base_dir, 'manifest')

    print('remote: %s (%s) %s' % (remote_name, remote_url, date_ymd))

    git_summary = {
        'remote_url': remote_url,
        'remote_name': remote_name,
        'commit': commit,
        'branch': git_current_branch(),
        'date': date,
    }

    file_list = git_file_list(path_list)
    # print(file_list)
    # assert False

    print('^' * 80)
    print('saving blame data')
    mkdir(data_dir)

    blame_map = BlameMap(git_summary, data_dir)
    if not force:
        blame_map.load()

    changed = update_blame_map(blame_map, file_list)
    if changed:
        blame_map.save()

    #  !@#$
    #
    # report_map = ReportMap(blame_map, path_list, reports_dir, file_list, author_list, ext_list)

    # get N top authors
    # blame_map2 = report_map.BlameMap  # optimization
    # for author in [all] + top:
    #     report_map = ReportMap(blame_map2, path_list, reports_dir, file_list, [author], ext_list)
    #     history = analyze_blame(report_map)
    #     save_tables(report_map)
    #     save_history(report_map, history, do_save, do_show)

    # This can be sped up be extracting
    # author_loc = Counter()              # {author: loc}
    # author_dates = {}                   # {author: (min date, max date)}
    # author_date_hash_loc = {}           # {author: [(date, hsh, loc)]}

    # to a separate print_function

    # e.g. get_author_stats()

    # assert author_list is None or isinstance(author_list, (set, list, tuple)), author_list


    report_map_all = ReportMap(git_summary, blame_map.catalog, path_list, reports_dir, file_list,
                               author_pattern, ext_pattern)
    author_loc_dates, top_authors = get_top_authors(report_map_all)
    top_a_list = [None] + [[a] for a in top_authors]
    reports_dir_list = []

    for a_list in top_a_list:
        rp = '[all]' if a_list is None else a_list[0]
        rp = os.path.join(reports_dir, rp)
        report_map = ReportMap(git_summary, report_map_all.catalog, path_list, rp,
                               None, None, None, a_list)

        history = analyze_blame(report_map)
        save_tables(report_map)
        save_history(report_map, history, do_save, do_show)
        print('reports_dir=%s' % os.path.abspath(report_map.reports_dir))
        reports_dir_list.append(os.path.abspath(report_map.reports_dir))

    print('+' * 80)
    print('base_dir=%s' % os.path.abspath(base_dir))
    for rd in reports_dir_list:
        print('reports_dir=%s' % rd)


if __name__ == '__main__':

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


    # author_list = {'Peter Williams'}  # !@#$
    # author_list = None
    # ext_list = {'.c', '.java', '.jsp', '.py', '.h', '.cpp', '.cxx', '.js'}

    do_save = True

    options, args = parser.parse_args()

    main(args, options.force, options.author_pattern, options.ext_pattern, do_save, options.do_show)
    print('DONE')
