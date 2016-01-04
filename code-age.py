# -*- coding: utf-8 -*-
"""
    Analyzes code age in a git repository

    Writes reports in the following locations

    e.g. For repository "cpython"

    [root]                                    Defaults to ~/git.stats
      ├── cpython                             Directory for https://github.com/python/cpython.git
      │   └── reports
      │       ├── 2011-03-06.d68ed6fc.2_0     Revision `d68ed6fc` which was created on 2011-03-06 on
      │       │   │                           branch `2.0`.
      │       │   └── __c.__cpp.__h           Report on *.c, *.cpp and *.h files in this revision
      │       │       ├── Guido_van_Rossum    Sub-report on author `Guido van Rossum`
      │       │       │   ├── code-age.png    Graph of code age. LoC / day vs date
      │       │       │   ├── code-age.txt    List of commits in the peaks in the code-age.png graph
      │       │       │   ├── details.csv     LoC in each directory in for these files and authors
      │       │       │   ├── newest-commits.txt      List of newest commits for these files and authors
      │       │       │   └── oldest-commits.txt      List of oldest commits for these files and authors
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
from scipy import signal
import pandas as pd
from pandas import Series, DataFrame, Timestamp
import matplotlib
import matplotlib.pylab as plt
from matplotlib.pylab import cycler
import bz2
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

# Python 2 / 3 stuff
PY2 = sys.version_info[0] < 3

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


#
# Configuration.
#
TIMEZONE = 'Australia/Melbourne'    # The timezone used for all commit times. TODO Make configurable
SHA_LEN = 8                         # The number of characters used when displaying git SHA-1 hashes
STRICT_CHECKING = False             # For validating code.
N_BLAME_PROCESSES = max(1, cpu_count() - 1)  # Number of processes to use for blaming
N_SHOW_THREADS = 8                 # Number of threads for running the many git show commands
DO_MULTIPROCESSING = True          # For test non-threaded peformance


# Set graphing style
matplotlib.style.use('ggplot')
plt.rcParams['axes.prop_cycle'] = cycler('color', ['b', 'y', 'k', '#707040', '#404070'])
plt.rcParams['savefig.dpi'] = 300

# Max length for file path names
try:
    PATH_MAX = os.pathconf(__file__, 'PC_NAME_MAX')
except:
    PATH_MAX = 255

# Files that we don't analyze. These are files that don't have lines of code so that blaming
#  doesn't make sense.
IGNORED_EXTS = {
    '.air', '.bin', '.bmp', '.cer', '.cert', '.der', '.developerprofile', '.dll', '.doc', '.docx',
    '.exe', '.gif', '.icns', '.ico', '.jar', '.jpeg', '.jpg', '.keychain', '.launch', '.pdf',
    '.pem', '.pfx', '.png', '.prn', '.so', '.spc', '.svg', '.swf', '.tif', '.tiff', '.xls', '.xlsx',
    '.tar', '.zip', '.gz', '.7z', '.rar',
    '.patch',
    '.dump'
   }


def _is_windows():
    """Returns: True if running on a MS-Windows operating system."""
    try:
        sys.getwindowsversion()
    except:
        return False
    else:
        return True

IS_WINDOWS = _is_windows()


if IS_WINDOWS:
    import win32api
    import win32process
    import win32con

    def lowpriority():
        """ Set the priority of the process to below-normal.
            http://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
        """
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)

else:
    def lowpriority():
        os.nice(1)


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


def truncate_sha(sha):
    """The way we show git SHA-1 hashes in reports."""
    return sha[:SHA_LEN]


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


def decode_to_str(bytes):
    """Decode byte list `bytes` to a unicode string trying utf-8 encoding first then latin-1.
    """
    if bytes is None:
        return None
    try:
        return bytes.decode('utf-8')
    except:
        return bytes.decode('latin-1')


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


def mkdir(path):
    """Create directory `path` including all intermediate-level directories and ignore
        "already exists" errors.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise


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
    """Returns: Weighted moving average of pandas Series `series` as a pandas Series.
        Weights are a triangle of width `window`.
        NOTE: If window is greater than the number of items in series then smoothing may not work
        well. See first few lines of function code.
    """
    if len(series) < 10:
        return series
    window = min(window, len(series))

    weights = np.empty(window, dtype=np.float)
    radius = (window - 1) / 2
    for i in range(window):
        weights[i] = radius + 1 - abs(i - radius)

    ma = np.convolve(series, weights, mode='same')
    assert ma.size == series.size, ([ma.size, ma.dtype], [series.size, series.dtype], window)

    sum_raw = series.sum()
    sum_ma = ma.sum()
    if sum_ma:
        ma *= sum_raw / sum_ma

    return Series(ma, index=series.index)


def procrustes(s, width=100):
    """Returns: String `s` fitted `width` or fewer chars, removing middle characters if necessary.
    """
    width = max(20, width)
    if len(s) > width:
        notch = int(round(width * 0.6)) - 5
        end = width - 5 - notch
        return '%s ... %s' % (s[:notch], s[-end:])
    return s


RE_EXT = re.compile(r'^\.\w+$')
RE_EXT_NUMBER = re.compile(r'^\.\d+$')


def get_ext(path):
    """Returns: extension of file `path`
    """
    parts = os.path.splitext(path)
    if not parts:
        ext = '[None]'
    else:
        ext = parts[-1]
        if not RE_EXT.search(ext) or RE_EXT_NUMBER.search(ext):
            ext = ''

    return ext


def exec_output(command, require_output):
    """Executes `command` which is a list of strings. If `require_output` is True then raise an
        exception is there is no stdout.
        Returns: The stdout of the child process as a string.
    """
    # TODO save stderr and print it on error
    try:
        output = subprocess.check_output(command)
    except:
        print('exec_output failed: command=%s' % ' '.join(command), file=sys.stderr)
        raise
    if require_output and not output:
        raise RuntimeError('exec_output: command=%s' % command)
    return decode_to_str(output)


def exec_output_lines(command, require_output):
    """Executes `command` which is a list of strings. If `require_output` is True then raise an
        exception is there is no stdout.
        Returns: The stdout of the child process as a list of strings, one string per line.
    """
    return exec_output(command, require_output).splitlines()


def exec_headline(command):
    """Execute `command` which is a list of strings.
        Returns: The first line stdout of the child process.
    """
    return exec_output(command, True).splitlines()[0]


def git_file_list(path_patterns=()):
    """Returns: List of files in current git revision matching `path_patterns`.
        This is basically git ls-files.
    """
    return exec_output_lines(['git', 'ls-files', '--exclude-standard'] + path_patterns, False)


def git_pending_list(path_patterns=()):
    """Returns: List of git pending files matching `path_patterns`.
    """
    return exec_output_lines(['git',  'diff', '--name-only'] + path_patterns, False)


def git_file_list_no_pending(path_patterns=()):
    """Returns: List of non-pending files in current git revision matching `path_patterns`.
    """
    file_list = git_file_list(path_patterns)
    pending = set(git_pending_list(path_patterns))
    return [path for path in file_list if path not in pending]


def git_diff(rev1, rev2):
    """Returns: List of files that differ in git revisions `rev1` and `rev2`.
    """
    return exec_output_lines(['git', 'diff', '--name-only', rev1, rev2], False)


def git_show_oneline(obj):
    """Returns: One-line description of a git object `obj`, which is typically a commit.
        https://git-scm.com/docs/git-show
    """
    return exec_headline(['git', 'show', '--oneline', '--quiet', obj])


def git_date(obj):
    """Returns: Date of a git object `obj`, which is typically a commit.
        NOTE: The returned date is standardized to timezone TIMEZONE.
    """
    date_s = exec_headline(['git', 'show', '--pretty=format:%ai', '--quiet', obj])
    return to_timestamp(date_s)


RE_REMOTE_URL = re.compile(r'(https?://.*/[^/]+(?:\.git)?)\s+\(fetch\)')

RE_REMOTE_NAME = re.compile(r'https?://.*/(.+?)(\.git)?$')


def git_remote():
    """Returns: The remote URL and a short name for the current repository.
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
    """Returns: git describe of current revision.
    """
    return exec_headline(['git', 'describe', '--always'])


def git_name():
    """Returns: git name of current revision.
    """
    return ' '.join(exec_headline(['git', 'name-rev', 'HEAD']).split()[1:])


def git_current_branch():
    """Returns: git name of current branch or None if there is no current branch (detached HEAD).
    """
    branch = exec_headline(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == 'HEAD':  # Detached HEAD?
        branch = None
    return branch


def git_current_revision():
    """Returns: SHA-1 of current revision.
    """
    return exec_headline(['git', 'rev-parse', 'HEAD'])


def git_revision_description():
    """Returns: Our best guess at describing the current revision"""
    description = git_current_branch()
    if not description:
        description = git_describe()
    return description


RE_PATH = re.compile(r'''[^a-z^0-9^!@#$\-+=_\[\]\{\}\(\)^\x7f-\xffff]''', re.IGNORECASE)
RE_SLASH = re.compile(r'[\\/]+')


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


def git_blame_text(path):
    """Returns: git blame text for file `path`
    """
    return exec_output(['git', 'blame', '-l', '-f', '-w', '-M', path], False)


RE_BLAME = re.compile(r'''
                      \^*([0-9a-f]{4,})\s+
                      .+?\s+
                      \(
                      (.+?)\s+
                      (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})
                      \s+(\d+)
                      \)''',
                      re.DOTALL | re.MULTILINE | re.VERBOSE)


def _debug_check_dates(max_date, sha_date_author, path_sha_loc):
    """Debug code to validate dates in `sha_date_author`, `path_sha_loc`
    """

    if not STRICT_CHECKING:
        return

    assert max_date <= MAX_DATE, max_date

    for path, sha_loc in path_sha_loc.items():
        for sha, loc in sha_loc.items():
            if loc <= 1:
                continue
            assert sha in sha_date_author, '%s not in sha_date_author' % [sha, path]
            date, _ = sha_date_author[sha]
            assert date <= max_date, ('date > max_date', sha, loc, [date, max_date], path)


class GitException(Exception):

    def __init__(self, msg=None):
        super(GitException, self).__init__(msg)
        self.git_msg = msg


if IS_WINDOWS:
    RE_LINE = re.compile(r'(?:\r\n|\n)+')
else:
    RE_LINE = re.compile(r'[\n]+')


def _extract_author_sha_loc(max_date, text, path):
    """Parses git blame output `text` and extracts LoC for each git hash found
        max_date: Latest valid date for a commit
        text: A string containing the git blame output of file `path`
        path: Path of blamed file. Used only for constructing error messages in this function
        Returns: sha_date_author, sha_loc
                 sha_date_author: {sha: (date, author)} over all SHA-1 hashes found in `text`
                 sha_loc: {sha: loc} over all SHA-1 hashes found in `text`. loc is "lines of code"
    """
    sha_date_author = {}
    sha_loc = Counter()

    lines = RE_LINE.split(text)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise GitException('is empty')

    for i, ln in enumerate(lines):
        if not ln:
            continue

        m = RE_BLAME.match(ln)
        if not m:
            raise GitException('bad line')
        if m.group(2) == 'Not Committed Yet':
            continue

        sha = m.group(1)
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
            raise GitException('bad date. sha=%s,date=%s' % (sha, date))

        sha_loc[sha] += 1
        sha_date_author[sha] = (date, author)

    if not sha_loc:
        raise GitException('is empty')

    return sha_date_author, sha_loc


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


class Persistable(object):
    """Base class that
        a) saves to disk,
                catalog: a dict of objects
                summary: a dict describing catalog
                manifest: a dict of sizes of objects in a catalog
        b) loads them from disk

        Derived classes must contain a dict data member called `TEMPLATE` that gives the keys of the
        data members to save / load and default constructors for each key.
    """

    @staticmethod
    def make_data_dir(path):
        return path_join(path, 'data')

    @staticmethod
    def update_dict(base_dict, new_dict):
        for k, v in new_dict.items():
            if k in base_dict:
                base_dict[k].update(v)
        return base_dict

    def _make_path(self, name):
        return path_join(self.base_dir, name)

    def __init__(self, summary, base_dir):
        """Initialize the data based on TEMPLATE and set summary to `summary`.

            summary: A dict giving a summary of the data to be saved
            base_dir: Directory that summary, data and manifest are to be saved to
        """
        assert 'TEMPLATE' in self.__class__.__dict__, 'No TEMPLATE in %s' % self.__class__.__dict__

        self.base_dir = base_dir
        self.data_dir = Persistable.make_data_dir(base_dir)
        self.summary = summary.copy()
        self.catalog = {k: v() for k, v in self.__class__.TEMPLATE.items()}

        for k, v in self.catalog.items():
            assert hasattr(v, 'update'), '%s.TEMPLATE[%s] does not have update(). type=%s' % (
                                          self.__class__.__name__, k, type(v))

    def load(self):
        catalog = load_object(self._make_path('data.pkl'), {})
        Persistable.update_dict(self.catalog, catalog)

        path = self._make_path('summary')
        if os.path.exists(path):
            self.summary = eval(open(path, 'rt').read())

    def save(self):
        # Load before saving in case another instance of this script is running
        path = self._make_path('data.pkl')
        if os.path.exists(path):
            catalog = load_object(path, {})
            self.catalog = Persistable.update_dict(catalog, self.catalog)

        # Save the data, summary and manifest
        mkdir(self.base_dir)
        save_object(path, self.catalog)
        open(self._make_path('summary'), 'wt').write(repr(self.summary))
        manifest = {k: len(v) for k, v in self.catalog.items()}
        open(self._make_path('manifest'), 'wt').write(repr(manifest))

    def __repr__(self):
        return repr([self.base_dir, {k: len(v) for k, v in self.catalog.items()}])


class BlameRepoState(Persistable):
    """Repository level persisted data structures
        Currently this is just sha_date_author.
    """
    TEMPLATE = {'sha_date_author': lambda: {}}


class BlameRevState(Persistable):
    """Revision level persisted data structures
        The main structure is path_sha_loc.
    """
    TEMPLATE = {
        'path_sha_loc': lambda: {},
        'path_set': lambda: set(),
        'bad_path_set': lambda: set(),
    }


def _task_extract_author_sha_loc(args):
    """Wrapper around _extract_author_sha_loc() to allow it to be executed by a multiprocessing
        Pool.
    """
    max_date, i, path = args
    sha_date_author, sha_loc, exception = None, None, None
    try:
        text = git_blame_text(path)
        sha_date_author, sha_loc = _extract_author_sha_loc(max_date, text, path)
    except Exception as e:
        exception = e
    return path, sha_date_author, sha_loc, exception


class BlameState(object):
    """A BlameState contains data from `git blame` that are used to compute reports.

        This data can take a long time to generate so we allow it to be saved to and loaded from
        disk so that it can be reused between runs.

        Data members: (All are read-only)
            repo_dir
            sha_date_author
            path_sha_loc
            path_set
            bad_path_set

        Typical usage:

            blame_state.load()                           # Load existing data from disk
            changed = blame_state.update_data(file_set)  # Blame files in file_set to update data
            if changed:
                blame_state.save()                        # Save updated data to disk

        Internal members: 'repo_dir', '_repo_state', '_rev_state', '_repo_base_dir'

        Disk storage
        ------------
        <repo_base_dir>             Defaults to ~/git.stats/<repository name>
            └── cache
                ├── 241d0c54        Data for revision 241d0c54
                │   ├── data.pkl    The data in a bzipped pickle.
                │   ├── manifest    Python file with dict of data keys and lengths
                │   └── summary     Python file with dict of summary date
                    ...
                ├
                ├── e7a3e5c4        Data for revision e7a3e5c4
                │   ├── data.pkl
                │   ├── manifest
                │   └── summary
                ├── data.pkl        Repository level data
                ├── manifest
                └── summary
    """
    def _debug_check(self):
        """Debugging code to check consistency of the data in a BlameState

        """
        if not STRICT_CHECKING:
            return
        for path, sha_loc in self.path_sha_loc.items():
            assert path in self.path_set, '%s not in self.path_set' % path
            for sha, loc in sha_loc.items():
                date, author = self.sha_date_author[sha]
        assert set(self.path_sha_loc.keys()) | self.bad_path_set == self.path_set, 'path sets wrong'

    def __init__(self, repo_base_dir, repo_summary, rev_summary):
        """

        repo_base_dir: Root of data saved for this repository. This is <repo_dir> in the storage
                       diagram. Typically ~/git.stats/<repository name>
        repo_summary = {
                'remote_url': remote_url,
                'remote_name': remote_name,
            }
        rev_summary = {
                'revision_sha': revision_sha,
                'branch': git_current_branch(),
                'description': description,
                'name': git_name(),
                'date': revision_date,
            }
        """
        self._repo_base_dir = repo_base_dir
        self._repo_dir = path_join(repo_base_dir, 'cache')
        self._repo_state = BlameRepoState(repo_summary, self.repo_dir)
        rev_dir = path_join(self._repo_state.base_dir,
                            truncate_sha(rev_summary['revision_sha']))
        self._rev_state = BlameRevState(rev_summary, rev_dir)

    def copy(self, rev_dir):
        """Returns: A copy of self with its rev_dir member replaced by `rev_dir`
        """
        blame_state = BlameState(self._repo_base_dir, self._repo_state.summary,
                                 self._rev_state.summary)
        blame_state._rev_state.base_dir = rev_dir
        return blame_state

    def load(self, max_date):
        """Loads a previously saved copy of it data from disk.
            Returns: A copy of self with its rev_dir member replaced by `rev_dir`
        """
        self._repo_state.load()
        self._rev_state.load()
        if STRICT_CHECKING:
            if max_date is not None:
                _debug_check_dates(max_date, self.sha_date_author, self.path_sha_loc)
            self._debug_check()
        return self

    def save(self):
        """Saves a copy of its data to disk
        """
        self._debug_check()
        self._repo_state.save()
        self._rev_state.save()
        if STRICT_CHECKING:
            self.load(None)

    def __repr__(self):
        return repr({k: repr(v) for k, v in self.__dict__.items()})

    @property
    def repo_dir(self):
        """Returns top directory for this repo's cached data.
            Typically ~/git.stats/<repository name>/cache
        """
        return self._repo_dir

    @property
    def sha_date_author(self):
        """Returns: {sha: (date, author)} for all commits that have been found in blaming this
            repository. sha is SHA-1 hash of commit
           This is a per-repository dict.
        """
        return self._repo_state.catalog['sha_date_author']

    @property
    def path_sha_loc(self):
        """Returns: {path: [(sha, loc)]} for all files that have been found in blaming this
            revision of this repository.
            path_sha_loc[path] is a list of (sha, loc) where sha = SHA-1 hash of commit and
            loc = lines of code from that commit in file `path`

           This is a per-revision dict.
        """
        return self._rev_state.catalog['path_sha_loc']

    @property
    def path_set(self):
        """Returns: set of paths of files that have been attempted to blame in this revision.
           This is a per-revision dict.
        """
        return self._rev_state.catalog['path_set']

    @property
    def bad_path_set(self):
        """Returns: set of paths of files that have been unsuccessfully attempted to blame in this
            revision.
            bad_path_set ∪ path_sha_loc.keys() == path_set
           This is a per-revision dict.
        """
        return self._rev_state.catalog['bad_path_set']

    def _get_peer_revisions(self):
        """Returns: data dicts of all revisions that have been blamed for this repository except
            this revision.
        """
        peer_dirs = (rev_dir for rev_dir in glob.glob(path_join(self.repo_dir, '*'))
                     if rev_dir != self._rev_state.base_dir and
                     os.path.exists(path_join(rev_dir, 'data.pkl')))

        for rev_dir in peer_dirs:
            rev = self.copy(rev_dir).load(None)
            yield rev_dir, rev

    def _update_from_existing(self, file_set):
        """Updates state of this repository / revision with data saved from blaming earlier
            revisions in this repository.
        """
        assert isinstance(file_set, set), type(file_set)

        # remaining_path_set = files in `file_set` that we haven't yet loaded
        remaining_path_set = file_set - self.path_set

        print('-' * 80)
        print('Update data from previous blames. %d remaining of %d files' % (
              len(remaining_path_set), len(file_set)))

        if not remaining_path_set:
            return

        peer_revisions = list(self._get_peer_revisions())
        print('Checking up to %d peer revisions for blame data' % len(peer_revisions))

        this_sha = self._rev_state.summary['revision_sha']
        this_date = self._rev_state.summary['date']

        peer_revisions.sort(key=lambda dir_rev: dir_rev[1]._rev_state.summary['date'])


        for i, (that_dir, that_rev) in enumerate(peer_revisions):

            if not remaining_path_set:
                break

            print('%2d: %s,' % (i, that_dir), end=' ')

            that_date = that_rev._rev_state.summary['date']
            sign = '>' if that_date > this_date else '<'
            print('%s %s %s' % (date_str(that_date), sign, date_str(this_date)), end=' ')

            # This is important. git diff can report 2 versions of a file as being identical while
            # git blame reports different commits for 1 or more lines in the file
            # In these cases we use the older commits.
            if that_date > this_date:
                print('later')
                continue

            that_sha = that_rev._rev_state.summary['revision_sha']
            that_path_set = that_rev.path_set
            that_bad_path_set = that_rev.bad_path_set
            diff_set = set(git_diff(this_sha, that_sha))

            self.bad_path_set.update(that_bad_path_set - diff_set)
            # existing_path_set: files in remaining_path_set that we already have data for
            existing_path_set = remaining_path_set & (that_path_set - diff_set)

            for path in existing_path_set:
                if path in that_rev.path_sha_loc:
                    self.path_sha_loc[path] = that_rev.path_sha_loc[path]
                    if STRICT_CHECKING:
                        for sha in self.path_sha_loc[path].keys():
                            assert sha in self.sha_date_author, '\n%s\nthis=%s\nthat=%s\n%s' % (
                                   (sha, path),
                                   self._rev_state.summary, that_rev._rev_state.summary,
                                   self.path_sha_loc[path])

                self.path_set.add(path)
                remaining_path_set.remove(path)
            print('%d files of %d remaining, diff=%d, used=%d' % (
                   len(remaining_path_set), len(file_set), len(diff_set), len(existing_path_set)))


    def _update_new_files(self, file_set, force):
        """Computes base statistics over whole revision
            Blames all files in `file_set`
            Updates: sha_date_author, path_sha_loc for files that are not already in path_sha_loc
        """

        rev_summary = self._rev_state.summary
        sha_date_author = self.sha_date_author
        path_set = self.path_set

        assert isinstance(sha_date_author, dict), type(sha_date_author)

        if not force:
            file_set = file_set - path_set
        n_files = len(file_set)
        print('-' * 80)
        print('Update data by blaming %d files' % len(file_set))

        loc0 = sum(sum(sha_loc.values()) for sha_loc in self.path_sha_loc.values())
        commits0 = len(sha_date_author)
        start = time.time()
        blamed = 0
        last_loc = loc0
        last_i = 0

        for path in file_set:
            path_set.add(path)
            if os.path.basename(path) in {'.gitignore'}:
                self.bad_path_set.add(path)
        file_set = file_set - self.bad_path_set

        # for i, path in enumerate(file_set):
        max_date = rev_summary['date']
        filenames = [(max_date, i, path) for i, path in enumerate(file_set)]
        filenames.sort(key=lambda dip: -os.path.getsize(dip[2]))

        with ProcessPool(ProcessPool.PROCESS, N_BLAME_PROCESSES) as pool:
            for i, (path, h_d_a, sha_loc, e) in enumerate(
                pool.imap_unordered(_task_extract_author_sha_loc, filenames)):

                if e is not None:

                    apath = os.path.abspath(path)
                    self.bad_path_set.add(path)

                    if isinstance(e, GitException):
                        if e.git_msg:
                            if e.git_msg != 'is empty':
                                print('    %s %s' % (apath, e.git_msg), file=sys.stderr)
                        else:
                            print('   %s cannot be blamed' % apath, file=sys.stderr)
                        continue
                    elif isinstance(e, subprocess.CalledProcessError):
                        if not os.path.exists(path):
                            print('    %s no longer exists' % apath, file=sys.stderr)
                            continue
                        elif os.path.isdir(path):
                            print('   %s is a directory' % apath, file=sys.stderr)
                            continue
                        elif stat.S_IXUSR & os.stat(path)[stat.ST_MODE]:
                            print('   %s is an executable. e=%s' % (apath, e), file=sys.stderr)
                            continue
                    raise

                self.sha_date_author.update(h_d_a)
                self.path_sha_loc[path] = sha_loc

                if i - last_i >= 100:
                    duration = time.time() - start

                    loc = sum(sum(sha_loc.values()) for sha_loc in self.path_sha_loc.values())
                    if loc != last_loc:
                        print('%d of %d files (%.1f%%), %d bad, %d LoC, %d commits, %.1f secs, %s' %
                              (i, n_files, 100 * i / n_files, i - blamed,
                              loc - loc0,
                              len(sha_date_author) - commits0,
                              duration,
                              procrustes(path, width=65)))
                        sys.stdout.flush()
                        last_loc = loc
                        last_i = i

                blamed += 1

                if STRICT_CHECKING:
                    _debug_check_dates(max_date, sha_date_author, self.path_sha_loc)
                    assert path in self.path_sha_loc, os.path.abspath(path)
                    assert self.path_sha_loc[path], os.path.abspath(path)
                    assert sum(self.path_sha_loc[path].values()), os.path.abspath(path)

        if STRICT_CHECKING:
            for path in file_set - self.bad_path_set:
                if os.path.basename(path) in {'.gitignore'}:
                    continue
                assert path in self.path_sha_loc, path

        print('~' * 80)
        duration = time.time() - start
        loc = sum(sum(sha_loc.values()) for sha_loc in self.path_sha_loc.values())
        print('%d files,%d blamed,%d lines,%d commits,%.1f seconds' % (len(file_set), blamed,
              loc, len(sha_date_author), duration))

    def update_data(self, file_set, force):
        """Updates blame state  for this revision for this repository over files in 'file_set'
            If `force` is True then blame all files in file_set, otherwise try tp re-use as much
            existing blame data as possible.

            Updates:
                repository: sha_date_author
                revision: path_sha_loc, file_set, bad_path_set
        """
        assert isinstance(file_set, set), type(file_set)
        n_paths0 = len(self.path_set)
        print('^' * 80)
        print('Update data for %d files. Previously %d files for this revision' % (
              len(file_set), n_paths0))
        self._debug_check()
        if not force:
            self._update_from_existing(file_set)
            self._debug_check()
        self._update_new_files(file_set, force)

        self._debug_check()

        if STRICT_CHECKING:
            for path in set(file_set) - self.bad_path_set:
                assert path in self.path_sha_loc, path

        return len(self.path_set) > n_paths0


def _filter_strings(str_iter, pattern):
    """Returns: Subset of strings in iterable `str_iter` that match regular expression `pattern`.
    """
    if pattern is None:
        return None
    assert isinstance(str_iter, set), type(str_iter)
    regex = re.compile(pattern, re.IGNORECASE)
    return {s for s in str_iter if regex.search(s)}


def filter_path_sha_loc(blame_state, path_sha_loc, file_set=None, author_set=None):
    """ blame_state: BlameState of revision
        path_sha_loc: {path: {sha: loc}} over all files in revisions
        file_set: files to filter on or None
        author_set: authors to filter on or None
    Returns: dict of items in `path_sha_loc` that match files in `file_set` and authors in
                `author_set`.
        NOTE: Does NOT modify path_sha_loc
    """
    assert file_set is None or isinstance(file_set, set), type(file_set)
    assert author_set is None or isinstance(author_set, set), type(author_set)

    if file_set:
        path_sha_loc = {path: sha_loc
                        for path, sha_loc in path_sha_loc.items()
                        if path in file_set}

    if STRICT_CHECKING:
        for path in path_sha_loc:
            assert path_sha_loc[path], path
            assert sum(path_sha_loc[path].values()), path

    if author_set:
        sha_set = {sha for sha, (_, author) in blame_state.sha_date_author.items()
                   if author in author_set}
        path_sha_loc = {path: {sha: loc for sha, loc in sha_loc.items()
                               if sha in sha_set}
                         for path, sha_loc in path_sha_loc.items()}
        path_sha_loc = {path: sha_loc
                        for path, sha_loc in path_sha_loc.items() if sha_loc}

    if STRICT_CHECKING:
        for path in path_sha_loc:
            assert path_sha_loc[path], path
            assert sum(path_sha_loc[path].values()), path

    return path_sha_loc


def _task_show_oneline(sha):
    """Wrapper around git_show_oneline() to allow it to be executed by a multiprocessing Pool.
    """
    text, exception = None, None
    try:
        text = git_show_oneline(sha)
    except Exception as e:
        exception = e
    return sha, text, exception


def parallel_show_oneline(sha_iter):
    """Run git_show_oneline() on SHA-1 hashes in `sha_iter` in parallel
        sha_iter: Iterable for some SHA-1 hashes
        Returns: {sha: text} over SHA-1 hashes sha in `sha_iter`. text is git_show_oneline output
    """
    sha_text = {}
    exception = None
    with ProcessPool(ProcessPool.THREAD, N_SHOW_THREADS) as pool:
        for sha, text, e in pool.imap_unordered(_task_show_oneline, sha_iter):
            if e:
                exception = e
                break
            sha_text[sha] = text

    if exception:
        raise exception
    return sha_text


def make_sha_path_loc(path_sha_loc):
    """path_sha_loc: {path: {sha: loc}}
       Returns: {sha: {path: loc}}
    """
    sha_path_loc = defaultdict(dict)
    for path, sha_loc in path_sha_loc.items():
        for sha, loc in sha_loc.items():
            sha_path_loc[sha][path] = loc
    return sha_path_loc


def make_report_name(default_name, components):
    if not components:
        return default_name
    elif len(components) == 1:
        return list(components)[0]
    else:
        return ('(%s)' % ','.join(sorted(components))) if components else default_name


def make_report_dir(base_dir, default_name, components, max_len=None):
    if max_len is None:
        max_len = PATH_MAX
    name = '.'.join(clean_path(cpt) for cpt in sorted(components)) if components else default_name
    return path_join(base_dir, name)[:max_len]


def get_totals(path_sha_loc):
    """Returns: total numbers of files, commits, LoC for files in `path_sha_loc`
    """
    all_commits = set()
    total_loc = 0
    for sha_loc in path_sha_loc.values():
        all_commits.update(sha_loc.keys())
        total_loc += sum(sha_loc.values())

    return len(path_sha_loc), len(all_commits), total_loc


def write_manifest(blame_state, path_sha_loc, report_dir, name_description, title):
    """Write a README file in `report_dir`
    """

    total_files, total_commits, total_loc = get_totals(path_sha_loc)
    totals = 'Totals: %d files, %d commits, %d LoC' % (total_files, total_commits, total_loc)

    repo_summary = blame_state._repo_state.summary
    rev_summary = blame_state._rev_state.summary
    details = 'Revision Details\n'\
              '----------------\n'\
              'Repository:  %s (%s)\n'\
              'Date:        %s\n'\
              'Description: %s\n'\
              'SHA-1 hash   %s\n' % (
                  repo_summary['remote_name'], repo_summary['remote_url'],
                  date_str(rev_summary['date']),
                  rev_summary['description'],
                  rev_summary['revision_sha'])

    with open(path_join(report_dir, 'README'), 'wt') as f:
        def put(s=''):
            f.write('%s\n' % s)

        put(title)
        put('=' * len(title))
        put()
        put(totals)
        put()
        if details:
            put(details)
            put()
        put('Files in this Directory')
        put('-----------------------')
        for name, description in sorted(name_description.items()):
            put('%-12s: %s' % (name, description))


def compute_tables(blame_state, path_sha_loc):
    """Compute summary tables over whole report showing number of files and LoC by author and
        file extension.

        blame_state: BlameState of revision
        path_sha_loc: {path: {sha: loc}} over files being reported
        Returns: df_author_ext_files, df_author_ext_loc where
            df_author_ext_files: DataFrame of file counts
            df_author_ext_loc:: DataFrame of file counts
            with both having columns of authors rows of file extensions.
    """

    sha_date_author = blame_state.sha_date_author

    exts = sorted({get_ext(path) for path in path_sha_loc.keys()})
    authors = sorted({author for _, author in sha_date_author.values()})

    assert '.patch' not in exts
    assert '.dump' not in exts

    author_ext_files = np.zeros((len(authors), len(exts)), dtype=np.int64)
    author_ext_loc = np.zeros((len(authors), len(exts)), dtype=np.int64)
    author_index = {author: i for i, author in enumerate(authors)}
    ext_index = {ext: i for i, ext in enumerate(exts)}

    if STRICT_CHECKING:
        for path, v in path_sha_loc.items():
            assert sum(v.values()), (path, len(v))
        for i, e in enumerate(exts):
            assert '--' not in e, e
            assert '.' not in e[1:], e

    for path, sha_loc in path_sha_loc.items():
        ext = get_ext(path)
        for sha, loc in sha_loc.items():
            _, author = sha_date_author[sha]
            a = author_index[author]
            e = ext_index[ext]
            author_ext_files[a, e] += 1
            author_ext_loc[a, e] += loc

    df_author_ext_files = DataFrame(author_ext_files, index=authors, columns=exts)
    df_author_ext_loc = DataFrame(author_ext_loc, index=authors, columns=exts)

    return df_author_ext_files, df_author_ext_loc


def get_tree_loc(path_loc):
    """ path_loc: {path: loc} over files in a git repository
        Returns: dir_tree_loc_frac for which
            dir_tree_loc_frac[path] = loc, frac where
            loc: LoC in path and all its descendants
            frac: loc / loc_parent where loc_parent is LoC in path's parent and all its descendants
    """

    dir_tree = defaultdict(set)  # {parent: children} over all directories parent that have children
    root = ''  # Root of git ls-files directory tree.

    for path in path_loc.keys():
        child = path
        while True:
            parent = os.path.dirname(child)
            if parent == child:
                root = parent
                break
            dir_tree[parent].add(child)
            child = parent

    # So we can index children below. See dir_tree[parent][i]
    dir_tree = {parent: list(children) for parent, children in dir_tree.items()}

    tree_loc = Counter()  # {path: loc} over all paths. loc = LoC in path and all its descendants if
                          # it is a directory

    # Traverse dir_tree depth first. Add LoC on leaf nodes then sum LoC over directories when
    # ascending
    stack = [(root, 0)]    # stack for depth first traversal of dir_tree
    while stack:
        parent, i = stack[-1]
        stack[-1] = parent, i + 1                       # Iterate over children
        if parent not in dir_tree:                      # Terminal node
            tree_loc[parent] = path_loc[parent]
            stack.pop()                                 # Ascend
        else:
            if i < len(dir_tree[parent]):               # Not done with children in this frame?
                stack.append((dir_tree[parent][i], 0))  # Descend
            else:                                       # Sum over descendants
                tree_loc[parent] = (path_loc.get(parent, 0) +
                                    sum(tree_loc[child] for child in dir_tree[parent]))
                stack.pop()                             # Ascend

    # dir_tree_loc is subset of tree_loc containing only directories (no normal files)
    dir_tree_loc = {path: loc for path, loc in tree_loc.items() if path in dir_tree and loc > 0}

    # dir_tree_loc_frac: {path: (loc, frac)} over all paths. loc = LoC in path and all its
    # descendants. frac = fraction of LoC path's parents and its descendants that are in path and
    # its descendants
    dir_tree_loc_frac = {path: (loc, 0) for path, loc in dir_tree_loc.items()}
    for parent, loc in dir_tree_loc.items():
        for child in dir_tree[parent]:
            if child in dir_tree_loc:
                dir_tree_loc_frac[child] = tuple((tree_loc[child], tree_loc[child] / loc))

    return dir_tree_loc_frac


def detailed_loc(path_sha_loc):
    """My attempt at showing the distribution of LoC over the directory structure of a git
        repository.

        I am using a table which seems unnatural but has the advantage that it can be viewed in
        powerful .csv table programs such as Excel.

        path_sha_loc: {path: {sha: loc}} over files in a git repository
        Returns: DataFrame with columns 'dir', 'LoC', 'frac' where
            dir: Sub-directories in the git repository
            LoC: LoC in dir and all its descendants
            frac: loc / loc_parent where loc_parent is LoC in dir's parent and all its descendants
    """

    path_loc = {path: sum(loc for _, loc in sha_loc.items())
                for path, sha_loc in path_sha_loc.items()}

    dir_tree_loc_frac = get_tree_loc(path_loc)
    dir_loc_frac = [(path, loc, frac) for path, (loc, frac) in dir_tree_loc_frac.items()]
    dir_loc_frac.sort()

    return DataFrame(dir_loc_frac, columns=['dir', 'LoC', 'frac'])


def save_summary_tables(blame_state, path_sha_loc, report_dir):
    """Save tables of number of files and LoC both by author and file extension to
        `author_ext_files.csv` and `author_ext_loc.csv' respectively.
    """

    if not path_sha_loc:
        print('No files to process')
        return

    def make_path(key):
        return path_join(report_dir, '%s.csv' % key)

    mkdir(report_dir)

    df_author_ext_files, df_author_ext_loc = compute_tables(blame_state, path_sha_loc)
    df_append_totals(df_author_ext_files).to_csv(make_path('author_ext_files'))
    df_append_totals(df_author_ext_loc).to_csv(make_path('author_ext_loc'))

    name_description = {
        'author_ext_files': 'Number of files of given extension in which author has code',
        'author_ext_loc': 'Number of LoC author in files of given extension by author'
    }

    title = 'Summary of File Counts and LoC by Author and File Extension'
    write_manifest(blame_state, path_sha_loc, report_dir, name_description, title)


DATE_INF_NEG = Timestamp('1911-11-22 11:11:11 -0700')
DATE_INF_POS = Timestamp('2111-11-22 11:11:11 -0700')


def get_top_authors(blame_state, path_sha_loc):
    """ blame_state: blame_state of current repository and revision
        author_report: files and authors to report on
        Returns: author_loc_dates, top_authors where
                author_loc_dates: {author: loc, min date, max date} over all author's commits
                top_authors: Authors in `author_report` in descending order of LoC
    """
    sha_date_author = blame_state.sha_date_author

    author_loc_dates = defaultdict(lambda: [0, DATE_INF_POS, DATE_INF_NEG])
    for sha_loc in path_sha_loc.values():
        for sha, loc in sha_loc.items():
            date, author = sha_date_author[sha]
            loc_dates = author_loc_dates[author]
            loc_dates[0] += loc
            loc_dates[1] = min(loc_dates[1], date)
            loc_dates[2] = max(loc_dates[2], date)

    assert author_loc_dates
    top_authors = sorted(author_loc_dates.keys(), key=lambda a: -author_loc_dates[a][0])
    return author_loc_dates, top_authors


def get_peak_commits(sha_loc, date_sha_loc, histo_peaks, window=20 * DAY):
    """ sha_loc: {sha: loc} over SHA-1 hashes in blame data
        date_sha_loc
        Returns: Lists of commits around `histo_peaks` which are peaks in a time series histogram
    """
    ts_histo, peak_ixy = histo_peaks
    dt = window / 2

    peak_ixy = sorted(peak_ixy, key=lambda k: _key_ixy_x(*k))

    peak_ends = [[x - dt, x + dt] for _, x, _ in peak_ixy]
    for i in range(1, len(peak_ends)):
        m0, m1 = peak_ends[i - 1]
        n0, n1 = peak_ends[i]
        if m1 > n0:
            peak_ends[i - 1][1] = peak_ends[i][0] = m1 + (n0 - m1) / 2

    peak_commits = []
    for (i, x, y), (m0, m1) in zip(peak_ixy, peak_ends):
        assert isinstance(x, pd.Timestamp), (type(x), x)
        this_sha_list = [sha for (date, sha, loc) in date_sha_loc if m0 <= date < m1]
        this_sha_list.sort(key=lambda sha: -sha_loc[sha])
        loc = sum(sha_loc[sha] for sha in this_sha_list)
        peak_commits.append((loc, x, this_sha_list))
    loc_total = sum(loc for loc, _, _ in peak_commits)

    return loc_total, list(zip(peak_ixy, peak_commits))


def find_peaks(ts_histo):
    """ ts_histo: A time series histogram
        Returns: Indexes of peaks in `ts_histo`
    """
    # TODO: Replace MIN_PEAK_HEIGHT with a calculated value

    MIN_PEAK_DAYS = 20   # !@#$ Reduce this
    MAX_PEAK_DAYS = 1
    MIN_PEAK_HEIGHT = 5  # !@#$ Depends on averaging Window

    # TODO: Tune np.arange(2, 10) * 10 to data
    width = delta_days(ts_histo.index.min(), ts_histo.index.max())
    width = min(max(width, 10), 100)
    peak_idx = signal.find_peaks_cwt(ts_histo, np.arange(2, 10) * width / 10)

    return [i for i in peak_idx
            if (delta_days(ts_histo.index[0], ts_histo.index[i]) >= MIN_PEAK_DAYS and
                delta_days(ts_histo.index[i], ts_histo.index[-1]) >= MAX_PEAK_DAYS and
                ts_histo.iloc[i] > MIN_PEAK_HEIGHT)
            ]


#
# Time series analysis
#  TODO: try different windows to get better peaks
def _compute_histogram_and_peaks(max_date, date_list, loc_list, window=60):
    """Returns: ts_histo, peak_idx

        loc: list of LoC events
        date: list of timestamps for loc
        window: width of weighted moving average window used to smooth data
        Returns: averaged time series, list of peaks in time series
                ts_histo: a histogram of LoC / day for events given by `loc` and `date`
                peak_idx: peaks in ts_histo
    """
    # ts_raw is histogram of LoC with 1 day bins. bins start at midnight on TIMEZONE
    # TODO: maybe offset dates in ts_raw to center of bin (midday)

    if STRICT_CHECKING:
        assert max(date_list) <= max_date, (max(date_list), max_date)

    ts_raw = Series(loc_list, index=date_list)  # TODO: dates may not be unique, guarantee this
    ts_days = ts_raw.resample('D', how='sum')  # 'D' = Day
    ts_days = ts_days.fillna(0)

    ts_histo = moving_average(ts_days, window) if window else ts_days
    peak_idx = find_peaks(ts_histo)
    return ts_histo, peak_idx


def make_histo_peaks(max_date, author_date_sha_loc, n_peaks, author_set=None):
    """ max_date: Max date allowed. For consistency checking.
        Returns: a histo_peaks (ts_histo, peak_ixy) for all authors in `author_set`
    """
    # TODO: Make date_loc a Series ?

    date_list, loc_list = [], []
    if author_set is None:
        author_set = author_date_sha_loc.keys()
    for author in author_set:
        dates, _, locs = zip(*author_date_sha_loc[author])
        date_list.extend(dates)
        loc_list.extend(locs)

    ts_histo, peak_idx = _compute_histogram_and_peaks(max_date, date_list, loc_list)

    peak_pxy = [(p, ts_histo.index[p], ts_histo.iloc[p]) for p in peak_idx]

    def key_pxy(p, x, y):
        return -y, x, p

    peak_pxy.sort(key=lambda k: key_pxy(*k))
    peak_ixy = [(i, x, y) for i, (p, x, y) in enumerate(peak_pxy[:n_peaks])]

    return ts_histo, tuple(peak_ixy)


def _key_ixy_x(i, x, y):
    """Key for sorting peak_ixy by x"""
    return x, i


def _get_xy_text(xy_plot, txt_width, txt_height):
    """Returns: positions of text labels for points `xy_plot`
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

        for i, ((y0, x0), (y, x), (y1, x1)) in enumerate(zip(yx_plot0, yx_plot, yx_plot1)):
            assert y0 <= y <= y1, (y0, y, y1)
        if not changed:
            break

    return [(x, y) for y, x in yx_plot]


def _plot_loc_date(ax, histo_peaks):
    """histo_peaks = ts_histo, peak_ixy
        Plot LoC vs date for time series `ts_histo` = histo_peaks[0].
        Optionally label peaks in `peak_ixy` = histo_peaks[1].
    """
    # TODO: Show areas
    ts_histo, peak_ixy = histo_peaks

    # FIXME. pandas plot() got a lot slower from python 2.7 to python 3.5
    ts_histo.plot(ax=ax)

    if not peak_ixy:
        return

    X0, X1, Y0, Y1 = ax.axis()

    peak_ixy = sorted(peak_ixy)  # Sort peak_ixy by y descending

    x0 = ts_histo.index[0]
    # TODO: Get actual text size. For now use the following guess.
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


def _plot_show(ax, blame_state, author_report, do_show, graph_path):
    """Show and/or save the current markings in matplotlib axes `ax`
    """

    repo_summary = blame_state._repo_state.summary
    rev_summary = blame_state._rev_state.summary
    path_sha_loc = author_report.path_sha_loc

    loc = sum(sum(sha_loc.values()) for sha_loc in path_sha_loc.values())

    ax.set_title('%s code age as of %s. files: %s, authors: %s\n'
                 'revision=%s, "%s", %d LoC' % (
                  repo_summary['remote_name'],
                  date_str(rev_summary['date']),
                  author_report.files_report_name,
                  author_report.report_name,
                  truncate_sha(rev_summary['revision_sha']),
                  rev_summary['description'],
                  loc
                 ))
    ax.set_xlabel('date')
    ax.set_ylabel('LoC / day')
    if graph_path:
        plt.savefig(graph_path)
    if do_show:
        print('Press q to quit, enter to continue...', end='', file=sys.stderr)
        plt.show(block=False)
        if PY2:
            answer = raw_input()
        else:
            answer = input()
        if answer.lower().startswith('q'):
            exit()


def plot_analysis(blame_state, author_report, histo_peaks, do_show, graph_path):

    # !@#$ update figure number
    fig, ax0 = plt.subplots(nrows=1)
    _plot_loc_date(ax0, histo_peaks)
    _plot_show(ax0, blame_state, author_report, do_show, graph_path)
    plt.close(fig)


def aggregate_author_date_sha_loc(author_date_sha_loc, author_set=None):
    """Get commit list [(date, sha, loc)] by author.
        author_date_sha_loc: {author: [(date, sha, loc)]} a dict of lists of (date, sha, loc) for
                               all commits by each author
        author_set: Authors to aggregate over.
        Returns: [(date, sha, loc)] for all authors in `author_set`
    """
    if author_set is None:
        author_set = author_date_sha_loc.keys()
    date_sha_loc = []
    for author in author_set:
        date_sha_loc.extend(author_date_sha_loc[author])
    return date_sha_loc


def put_newest_oldest(f, author, sha_path_loc, date_sha_loc, sha_date_author, n_commits,
    n_files, is_newest):
    """Write a report of oldest surviving commits in current revision
        n_commits: Max number of commits to write
        n_files: Max number of files to write for each revision

        author: Date of oldest revision
        commit 1: Same format as write_legend
            file 1: LoC
            file 2: LoC
            ...
        commit 2: ...
            ...
    """

    def put(s):
        f.write('%s\n' % s)

    def date_key(date, sha, loc):
        return date, loc

    date_sha_loc = sorted(date_sha_loc, key=lambda k: date_key(*k))
    if is_newest:
        date_sha_loc.reverse()
    loc_total = sum(loc for _, _, loc in date_sha_loc)

    put('=' * 80)
    put('%s: %d commits %d LoC' % (author, len(date_sha_loc), loc_total))

    cnt = 0
    sha_text = parallel_show_oneline((sha for _, sha, _ in date_sha_loc[:n_commits]))

    for i, (date, sha, loc) in enumerate(date_sha_loc[:n_commits]):
        put('.' * 80)
        put('%5d LoC, %s %s' % (loc, date_str(sha_date_author[sha][0]),
            sha_text[sha]))

        path_loc = sorted(sha_path_loc[sha].items(), key=lambda k: (-k[1], k[0]))
        for j, (path, l) in enumerate(path_loc[:n_files]):
            put('%5d LoC,   %s' % (l, path))
        cnt += 1


class AuthorReport(object):
    """AuthorReport's contain data for reports. Unlike BlameMaps they don't make git calls, they
        only filter day, write reports and plot graphs so they _should_ be fast.

        Members:
            report_dir: Directory that report is to be saved to:
            path_sha_loc: {path: {sha: loc}} over paths of all files in report where
                           {sha: loc} is a dict of LoC for each SHA-1 hash that is in the blame
                           of `path`
            author_set: Authors whose code is to included in report
    """

    def __init__(self, blame_state, path_sha_loc, max_date, files_report_name, files_report_dir,
        author_set):

        assert author_set is None or isinstance(author_set, set), author_set
        assert files_report_dir

        self.blame_state = blame_state
        self.report_name = make_report_name('[all-authors]', author_set)
        self.report_dir = make_report_dir(files_report_dir, '[all-authors]', author_set,
                                          PATH_MAX - 20)
        self.files_report_name = files_report_name
        self.max_date = max_date
        self.author_set = author_set
        self.path_sha_loc = filter_path_sha_loc(blame_state, path_sha_loc, author_set=author_set)
        self.name_description = {}

        if STRICT_CHECKING:
            for path in path_sha_loc:
                assert path_sha_loc[path], path
                assert sum(path_sha_loc[path].values()), path

            assert path_sha_loc, author_set
            _debug_check_dates(max_date, blame_state.sha_date_author, path_sha_loc)
            assert ':' not in self.report_dir[2:], self.report_dir

    def write_legend(self, legend_path, histo_peaks, n_top_commits):

        loc_auth, peak_ixy_commits = get_peak_commits(self.sha_loc, self.date_sha_loc, histo_peaks)
        peak_ixy_commits.sort(key=lambda k: _key_ixy_x(*k[0]))

        sha_gen = (sha for _, (_, _, this_sha_list) in peak_ixy_commits
                   for sha in this_sha_list[:n_top_commits])

        sha_text = parallel_show_oneline(sha_gen)

        with open(legend_path, 'wt') as f:

            def put(s):
                f.write('%s\n' % s)

            put('=' * 80)
            put('%s: %d peaks %d LoC' % (self.report_name, len(peak_ixy_commits), loc_auth))

            for (i, x, y), (loc, peak, this_sha_list) in peak_ixy_commits:
                put('.' * 80)
                put('%3d) %d commits %d LoC around %s' % (i + 1, len(this_sha_list), loc,
                    date_str(peak)))
                for sha in sorted(this_sha_list[:n_top_commits],
                                  key=lambda k: self.blame_state.sha_date_author[k][0]):
                    put('%5d LoC, %s %s' % (self.sha_loc[sha],
                        date_str(self.blame_state.sha_date_author[sha][0]), sha_text[sha]))

    def write_newest(self, newest_path, sha_path_loc, n_commits, n_files):
        with open(newest_path, 'wt') as f:
            put_newest_oldest(f, self.report_name, sha_path_loc, self.date_sha_loc,
                              self.blame_state.sha_date_author, n_commits, n_files, True)

    def write_oldest(self, oldest_path, sha_path_loc, n_commits, n_files):
        with open(oldest_path, 'wt') as f:
            put_newest_oldest(f, self.report_name, sha_path_loc, self.date_sha_loc,
                              self.blame_state.sha_date_author, n_commits, n_files, False)

    def analyze_blame(self):
        """Compute derived member variables
            self.sha_loc               {sha: loc} over all commit hashes in self.path_sha_loc
            self.author_date_sha_loc   {author: [(date, sha, loc)]} for all author's commits
        """
        sha_date_author = self.blame_state.sha_date_author
        path_sha_loc = self.path_sha_loc

        sha_loc = Counter()                # {sha: loc} over all commit hashes in path_sha_loc
        for h_l in path_sha_loc.values():
            for sha, loc in h_l.items():
                sha_loc[sha] += loc

        author_date_sha_loc = defaultdict(list)     # {author: [(date, sha, loc)]}
        for sha, loc in sha_loc.items():
            date, author = sha_date_author[sha]
            author_date_sha_loc[author].append((date, sha, loc))

        for author in author_date_sha_loc.keys():
            author_date_sha_loc[author].sort()

        self.sha_loc = sha_loc
        self.author_date_sha_loc = author_date_sha_loc

    def save_code_age(self, do_save, do_show, n_peaks, n_top_commits, n_newest_oldest, n_files,
        n_min_days):
        """Create a graph (time series + markers)
            a list of commits in for each peak
            + n biggest commits
            <name>.png
            <name>.txt
        """
        blame_state = self.blame_state

        mkdir(self.report_dir)

        self.date_sha_loc = aggregate_author_date_sha_loc(self.author_date_sha_loc,
                                                          self.author_set)

        histo_peaks = make_histo_peaks(self.max_date, self.author_date_sha_loc, n_peaks,
                                       self.author_set)

        ts_histo, _ = histo_peaks
        if ts_histo.max() - ts_histo.min() < n_min_days:
            print('%d days of activity < %d. Not reporting for %s' % (
                  ts_histo.max() - ts_histo.min(), n_min_days, self.report_name))
            return False

        if do_save:
            graph_path = path_join(self.report_dir, 'code-age.png')
            legend_path = path_join(self.report_dir, 'code-age.txt')
            newest_path = path_join(self.report_dir, 'newest-commits.txt')
            oldest_path = path_join(self.report_dir, 'oldest-commits.txt')

            sha_path_loc = make_sha_path_loc(blame_state.path_sha_loc)
            self.write_legend(legend_path, histo_peaks, n_top_commits)
            self.write_newest(newest_path, sha_path_loc, n_newest_oldest, n_files)
            self.write_oldest(oldest_path, sha_path_loc, n_newest_oldest, n_files)

            self.name_description['code-age.png'] = 'Graph of code age'
            self.name_description['code-age.txt'] = 'Commits around peaks in code-age.png'
            self.name_description['newest-commits.txt'] = 'Newest commits'
            self.name_description['oldest-commits.txt'] = 'Oldest surviving commits'

        else:
            graph_path = None

        plot_analysis(blame_state, self, histo_peaks, do_show, graph_path)
        return True

    def save_details_table(self):
        """Writes a table showing the distribution of LoC over the directory structure of a git
            repository to 'details.csv'

            See detailed_loc() for the structure of the table.

        """
        if not self.path_sha_loc:
            print('No files to process')

        details_path = path_join(self.report_dir, 'details.csv')
        df_dir_tree_loc = detailed_loc(self.path_sha_loc)
        df_dir_tree_loc.to_csv(details_path)
        self.name_description['details.csv'] = 'Distribution of LoC over the directory structure'
        print('Details: %s' % details_path)

    def write_manifest(self):
        total_loc, total_commits, total_files = get_totals(self.path_sha_loc)
        title = 'Code Age Report for files %s and authors %s.' % (
                    self.files_report_name, self.report_name)
        write_manifest(self.blame_state, self.path_sha_loc, self.report_dir, self.name_description,
                       title)


def _show_summary(repo_summary, rev_summary):
    all_summary = repo_summary.copy()
    all_summary.update(rev_summary)

    # description and name are usually the same but sometimes they differ like this
    #     branch: None
    #     description: svn-branch/14-0-fixes
    #     name: tags/svn-branch/14-0-fixes^0
    print('=' * 80)
    for k, v in sorted(all_summary.items()):
        print('%15s: %s' % (k, v))
    print('-' * 80)


def filter_bad_files(blame_state, file_set):
    """ blame_state: BlameState of this revision
        file_set: Some set of files in the blame_state
        Returns: `file_set` with the bad files filtered out. See BlameState._update_new_files()
                 for what bad files are.
    """
    if blame_state.bad_path_set:
        good_file_set = file_set - blame_state.bad_path_set
        print('`' * 80)
        bad_files_display = [path for path in blame_state.bad_path_set & file_set
                             if os.path.basename(path) not in {'.gitignore'}]
        print('%d bad files' % len(bad_files_display))
        for i, path in enumerate(sorted(bad_files_display)):
            print('%3d: %s' % (i, os.path.abspath(path)))
        print('"' * 80)

        print('file_set=%d (%d bad)' % (len(good_file_set),
              len(blame_state.bad_path_set & file_set)))
        file_set = good_file_set
    return file_set


def create_files_reports(gitstatsignore, path_patterns, do_save, do_show, force, author_pattern,
    n_top_authors, n_peaks, n_top_commits, n_newest_oldest, n_files, n_min_days):
    """Creates a set of reports for a specified pattern of files in the current revision of the git
        repository this script is being run from.

        Creates one report for all authors and one for each author individually (for all
        authors that match the `author_pattern`, `n_top_authors` criteria)

    e.g. For repository "cpython" revision "d68ed6fc"

    [root]                                    Defaults to ~/git.stats
      ├── cpython                             Directory for https://github.com/python/cpython.git
      │   └── reports
      │       ├── 2011-03-06.d68ed6fc.2_0     Revision "d68ed6fc" which was created on 2011-03-06 on
      │       │   │                           on branch "2.0"
      │       │   └── __c.__cpp.__h           Reports on *.c, *.cpp and *.h files in this revision
      │       │       ├── Guido_van_Rossum    Report on author `Guido van Rossum`
      │       │       │   ├── code-age.png    Graph of code age. LoC / day vs date
      │       │       │   ├── code-age.txt    List of commits in the code-age.png graph peaks
      │       │       │   ├── details.csv     LoC in each directory
      │       │       │   ├── newest-commits.txt      List of newest commits
      │       │       │   └── oldest-commits.txt      List of oldest commits
                          ... More reports for other authors and one for [all-authors]
                      ... More groups of reports for other file patterns
                   ... More groups of groups of reports for other revisions
           ... More subtrees for other repositories

    Params:
        gitstatsignore: File with path patterns to ignore. If None, use 'gitstatsignore' in
                        current directory
        path_patterns: File patterns to pass to git ls-files. The resulting files (called
                       `file_set` through the code) are reported.
        do_save: Save results to disk?
        do_show: Show graphs interactively?
        force: Force re-blaming of files in `file_set`.
        author_pattern: Regex pattern to filter authors by.
        n_top_authors: Number of authors to create reports for. Author reports are created in
                       descending order of LoC of that author.
        n_peaks: Number of peaks to find in a code age graph.
        n_top_commits: Number of top commits to list for each autho r. These are the highest peaks
                       in the smoothed histogram of commits in the code-age.png graph in
                       descending order of LoC
        n_newest_oldest: Number of commits to show in newest-commits.txt and oldest-commits.txt
        n_files: Number of fies to show for each commit in newest-commits.txt and oldest-commits.txt
        n_min_days: Minimum number of days of commits required for an author to be reported on
    """

    remote_url, remote_name = git_remote()
    description = git_revision_description()
    revision_sha = git_current_revision()
    revision_date = git_date(revision_sha)

    # Set up directory structure described in function docstring
    root_dir = path_join(os.path.expanduser('~'), 'git.stats')
    repo_base_dir = path_join(root_dir, remote_name)
    repo_dir = path_join(repo_base_dir, 'reports')
    rev_dir = path_join(repo_dir, '.'.join([date_str(revision_date), truncate_sha(revision_sha),
                                            clean_path(description)]))
    path_patterns = [normalize_path(path) for path in path_patterns]

    ppatterns = None if (len(path_patterns) == 1 and path_patterns[0] == '.') else path_patterns
    files_report_name = make_report_name('[all-files]', ppatterns)
    files_report_dir = make_report_dir(rev_dir, '[all-files]', ppatterns, PATH_MAX - 50)

    # TODO: Add a branches file in rev_dir
    repo_summary = {
        'remote_url': remote_url,
        'remote_name': remote_name,
    }
    rev_summary = {
        'revision_sha': revision_sha,
        'branch': git_current_branch(),
        'description': description,
        'name': git_name(),
        'date': revision_date,
    }
    _show_summary(repo_summary, rev_summary)

    # Compute `file_set`, the files to be reported on
    file_set0 = set(git_file_list_no_pending(path_patterns))
    file_set = file_set0 - get_ignored_files(gitstatsignore)
    file_set = {path for path in file_set if get_ext(path) not in IGNORED_EXTS}
    print('path_patterns=%s' % path_patterns)
    print('file_set=%d total, %d excluding ignored' % (len(file_set0), len(file_set)))

    if not file_set:
        print('path_patterns=%s selects no files. Nothing to do' % path_patterns)
        return

    blame_state = BlameState(repo_base_dir, repo_summary, rev_summary)
    if not force:
        blame_state.load(revision_date)
    changed = blame_state.update_data(file_set, force)
    if changed:
        blame_state.save()

    if STRICT_CHECKING:
        for path in blame_state.path_sha_loc:
            assert blame_state.path_sha_loc[path], os.path.abspath(path)
            assert sum(blame_state.path_sha_loc[path].values()), os.path.abspath(path)

    file_set = filter_bad_files(blame_state, file_set)
    if not file_set:
        print('Only bad files. Nothing to do' % path_patterns)
        return

    all_authors = {author for _, author in blame_state.sha_date_author.values()}
    all_author_set = _filter_strings(all_authors, author_pattern)

    # path_sha_loc is for files matched by file_set and author_set
    path_sha_loc = filter_path_sha_loc(blame_state, blame_state.path_sha_loc,
                                       file_set=file_set, author_set=all_author_set)

    save_summary_tables(blame_state, path_sha_loc, files_report_dir)

    author_loc_dates, top_authors = get_top_authors(blame_state, path_sha_loc)

    author_report_list = []

    for author in [None] + top_authors[:n_top_authors]:
        author_set = None if author is None else {author}
        author_report = AuthorReport(blame_state, path_sha_loc, revision_date, files_report_name,
                                     files_report_dir, author_set)
        author_report.analyze_blame()
        saved = author_report.save_code_age(do_save, do_show, n_peaks, n_top_commits,
                                            n_newest_oldest, n_files, n_min_days)
        if not saved:
            continue

        author_report.save_details_table()
        author_report.write_manifest()
        author_report_list.append(os.path.abspath(author_report.report_dir))
        sys.stdout.flush()


    print('+' * 80)
    print('%2d reports directories' % len(author_report_list))
    for i, author_dir in enumerate(author_report_list):
        print('%2d: %s' % (i, author_dir))
    print('    %s' % os.path.abspath(files_report_dir))
    print('description="%s"' % description)


def main():
    import optparse

    lowpriority()

    parser = optparse.OptionParser('python ' + sys.argv[0] + ' [options] [<directory>]')
    parser.add_option('-c', '--code-only', action='store_true', default=False, dest='code_only',
                      help='Show only code files')
    parser.add_option('-f', '--force', action='store_true', default=False, dest='force',
                      help='Force running git blame over source code')
    parser.add_option('-s', '--show', action='store_true', default=False, dest='do_show',
                      help='Pop up graphs as we go')
    parser.add_option('-a', '--authors', dest='author_pattern', default=None,
                      help='Analyze only code with these authors')
    parser.add_option('-g', '--gitstatsignore', dest='gitstatsignore', default=None,
                      help='File patterns to ignore')

    # Display / Report options
    parser.add_option('-A', '--number-top-authors', dest='n_top_authors', type=int, default=20,
                      help='Number of authors to list')
    parser.add_option('-P', '--number-peaks', dest='n_peaks', type=int, default=10,
                      help='Number of peaks to find in a code age graph')
    parser.add_option('-O', '--number-oldest-commits', dest='n_newest_oldest', type=int, default=20,
                      help='Number of oldest (and newest) commits to list')
    parser.add_option('-C', '--number-commits', dest='n_top_commits', type=int, default=5,
                      help='Number of commits to list for each author')
    parser.add_option('-F', '--number-files', dest='n_files', type=int, default=3,
                      help='Number of files to list for each commit in a report')
    parser.add_option('-D', '--min-days', dest='n_min_days', type=int, default=1,
                      help='Minimum days of blamed code to report')

    do_save = True

    options, args = parser.parse_args()

    create_files_reports(options.gitstatsignore,
                         args, do_save, options.do_show, options.force,
                         options.author_pattern,
                         options.n_top_authors, options.n_peaks,
                         options.n_top_commits, options.n_newest_oldest, options.n_files,
                         options.n_min_days)

if __name__ == '__main__':
    print('N_BLAME_PROCESSES=%d' % N_BLAME_PROCESSES)
    main()
    print()
