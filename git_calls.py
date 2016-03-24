# -*- coding: utf-8 -*-
"""
    Wrapper around git command line interface

"""
from __future__ import division, print_function
import subprocess
import sys
import time
import re
import os
from collections import namedtuple


# Python 2 / 3 stuff
PY2 = sys.version_info[0] < 3
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
STRICT_CHECKING = True              # For validating code.


ExecResult = namedtuple('ExecResult', ['command', 'ret', 'out', 'err'])


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


def truncate_sha(sha):
    """The way we show git SHA-1 hashes in reports."""
    return sha[:SHA_LEN]


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


def run_program(command, async_timeout=60.0):
    # r = subprocess.run(command)
    # print('$' * 80)
    # print(r)
    # for a in r.args:
    #     print(a)
    ps = subprocess.Popen(command,
                          # shell=True
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    out, err = ps.communicate()[:2]

    t0 = time.time()
    while True:
        ret = ps.poll()
        if ret is not None or time.time() <= t0 + async_timeout:
            break
        time.sleep(1.0)


    return ret, out, err


def border(name):
    return '%s: %s' % (name, (78 - len(name)) * '^')

BORDER = '^' * 80
BORDER_FAILURE = '*' * 80


def format_error(exec_result, name=None, failure=None):
    command, ret, out, err = exec_result

    assert name is None or isinstance(name, str), (name, type(name))
    assert isinstance(ret, int), (ret, type(ret))

    if name is None:
        name = ' '.join(command)

    summary = '%s: ret=%d,out=%d,err=%d' % (name, ret, len(out), len(err))
    parts = [border('out'), out, border('err'), err]
    if failure:
        parts.extend([BORDER_FAILURE, failure])
    parts.extend([BORDER, summary])

    return '\n%s\n' % '\n'.join(parts)


_git_debugging = False


def git_debugging(debugging=None):
    global _git_debugging
    current_debugging = _git_debugging
    if debugging is not None:
        _git_debugging = debugging
    return current_debugging


def exec_output(command, require_output):
    """Executes `command` which is a list of strings. If `require_output` is True then raise an
        exception is there is no stdout.
        Returns: ret, output_str, error_str
            ret: return code from exec'd process
            output_str: stdout of the child process as a string
            error_str: stderr of the child process as a string
    """
    exception = None
    output = None
    error = None
    ret = -1

    if _git_debugging:
        print('exec_output(%s)' % ' '.join(command))

    try:
        ret, output, error = run_program(command)
    except Exception as e:
        exception = e

    if exception is None and require_output and not output:
        exception = RuntimeError('exec_output: command=%s' % command)

    output_str = decode_to_str(output) if output is not None else ''
    error_str = decode_to_str(error) if error is not None else ''
    exec_result = ExecResult(command, ret, output_str, error_str)

    if exception is not None:
        format_error(exec_result)
        raise exception

    return exec_result


def exec_output_lines(command, require_output, separator=None):
    """Executes `command` which is a list of strings. If `require_output` is True then raise an
        exception is there is no stdout.
        Returns: ret, output_str, error_str
            ret: return code from exec'd process
            output_lines: stdout of the child process as a list of strings, one string per line
            error_str: stderr of the child process as a string
    """
    exec_result = exec_output(command, require_output)
    if separator is not None:
        output_lines = exec_result.out.split(separator)
    else:
        output_lines = exec_result.out.splitlines()
    return output_lines, exec_result


def exec_headline(command):
    """Execute `command` which is a list of strings.
        Returns: The first line stdout of the child process.
        Returns: ret, output_str, error_str
            ret: return code from exec'd process
            output_line: he first line stdout of the child process.
            error_str: stderr of the child process as a string
    """
    output_lines, exec_result = exec_output_lines(command, True)
    return output_lines[0], exec_result


def git_config_set(key, value):
    return exec_output(['git', 'config', key, value], False)


def git_config_unset(key):
    return exec_output(['git', 'config', '--unset', key], False)


def git_reset(obj, hard=True):
    command = ['git', 'reset']
    if hard:
        command.append('--hard')
    command.append(obj)
    return exec_output(command, False)


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


def git_log(pattern):
    """Returns: List of commits in ancestors of current git revision with commit messages matching
        `pattern`.
        This is basically git log --grep=<pattern>.
    """
    # git ls-files -z returns a '\0' separated list of files terminated with '\0\0'
    command = ['git', 'log', '-z',
     '--perl-regexp',
     # '--grep="%s"' % pattern
     "--grep='%s'" % pattern
    ]

    bin_list, exec_result = exec_output_lines(command, False, '\0')
    print('bin_list=%d' % len(bin_list))
    print(exec_result)

    commit_list = []
    for commit in bin_list:
        if not commit:
            break
        commit_list.append(commit)

    return commit_list, exec_result


def git_show(obj=None, quiet=False):
    """Returns: Description of a git object `obj`, which is typically a commit.
        https://git-scm.com/docs/git-show
    """
    command = ['git', 'show']
    if quiet:
        command.append('--quiet')
    if obj is not None:
        command.append(obj)

    return exec_output(command, True)


def git_show_oneline(obj=None):
    """Returns: One-line description of a git object `obj`, which is typically a commit.
        https://git-scm.com/docs/git-show
    """
    command = ['git', 'show', '--oneline', '--quiet']
    if obj is not None:
        command.append(obj)

    return exec_headline(command)


def git_show_sha(obj=None):
    """Returns: SHA-1 hash `obj`, which is typically a commit.
        https://git-scm.com/docs/git-show
    """
    command = ['git', 'show', '--format=%H']
    if obj is not None:
        command.append(obj)
    return exec_headline(command)

if False:
    ret, line, err = git_show_sha()
    print('ret=%d,err="%s"' % (ret, err))
    print(line)
    assert False


# def git_date(obj):
#     """Returns: Date of a git object `obj`, which is typically a commit.
#         NOTE: The returned date is standardized to timezone TIMEZONE.
#     """
#     date_s = exec_headline(['git', 'show', '--pretty=format:%ai', '--quiet', obj])
#     return to_timestamp(date_s)


RE_REMOTE_URL = re.compile(r'(https?://.*/[^/]+(?:\.git)?)\s+\(fetch\)')
RE_REMOTE_NAME = re.compile(r'https?://.*/(.+?)(\.git)?$')


def git_remote():
    """Returns: The remote URL and a short name for the current repository.
    """
    # $ git remote -v
    # origin  https://github.com/FFTW/fftw3.git (fetch)
    # origin  https://github.com/FFTW/fftw3.git (push)

    try:
        output_lines, exec_result = exec_output_lines(['git', 'remote', '-v'], True)
    except Exception as e:
        print('git_remote error: %s' % e)
        return None, None

    for line in output_lines:
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
    branch, exec_result = exec_headline(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == 'HEAD':  # Detached HEAD?
        branch = None
    return branch, exec_result


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


def git_blame_text(path):
    """Returns: git blame text for file `path`
    """
    return exec_output(['git', 'blame', '-l', '-f', '-w', '-M', path], False)

RE_REMOTE = re.compile(r'origin/(.*?)\s*$')
RE_LOCAL = re.compile(r'\*?\s*(.*?)\s*$')


def git_track_all(delete_local):
    """Track all remote branches
        If `delete_local` is True then delete all local branches (except checked out branch)
        before tracking.
        Returns: local_branches, exec_result
            local_branches: List of local branches with remotes
            exec_result: Result of exec_output()
    """

    local_branches0, exec_result = exec_output_lines(['git', 'branch'], False)
    assert exec_result.ret == 0, format_error(exec_result)
    local_branches = set()
    for i, local in enumerate(local_branches0):
        m = RE_LOCAL.search(local)
        assert m, 'local="%s"' % local
        local = m.group(1)
        local_branches.add(local)

    current_branch, exec_result = git_current_branch()
    assert exec_result.ret == 0, format_error(exec_result)

    if delete_local:
        removed_branches = set()
        for local in local_branches:
            if local == current_branch:
                continue
            exec_result = exec_output(['git', 'branch', '-D', local], False)
            assert exec_result.ret == 0, format_error(exec_result)
            removed_branches.add(local)
        local_branches -= removed_branches

    remote_branches0, exec_result = exec_output_lines(['git', 'branch', '-r'], False)
    assert exec_result.ret == 0, format_error(exec_result)

    local_remote = {}
    for i, remote in enumerate(remote_branches0):
        if 'origin/HEAD ->' in remote:
            continue
        m = RE_LOCAL.search(remote)
        assert m, remote
        remote = m.group(1)
        m = RE_REMOTE.search(remote)
        assert m, remote
        local = m.group(1)
        local_remote[local] = remote

    for i, (local, remote) in enumerate(sorted(local_remote.items())):

        if local in local_branches:
            continue

        print('%3d: Tracking "%s" from "%s"' % (i, local, remote))

        exec_result = exec_output(['git', 'branch', '--track', local], False)
        if exec_result.ret != 0:
            print('Could not track remote branch local="%s",remote="%s"' % (local, remote))
        assert exec_result.ret == 0, format_error(exec_result)
        local_branches.add(local)

    # Only interested in branches that have a remote
    local_branches = {local: remote for local in local_branches if local in local_remote}

    return local_branches, exec_result


def git_fetch():
    """
    """
    return exec_output(['git', 'fetch', '--all', '--tags', '--prune', '--force'], False)


def git_checkout(branch):
    """
    """
    return exec_output(['git', 'checkout', branch], False)


def _find_conflicts(out_lines):
    return [line for line in out_lines if line.startswith('CONFLICT')]


def git_pull(branch):
    """
    """
    out_lines, exec_result = exec_output_lines(['git', 'pull', '--force', '--ff', 'origin', branch],
                                               False)
    return _find_conflicts(out_lines), out_lines, exec_result


def git_push(branch):
    """
    """
    assert False, branch
    return exec_output(['git', 'push', 'origin', branch], False)


def git_merge(branch):
    """
    """
    out_lines, exec_result = exec_output_lines(['git', 'merge', branch], False)
    return _find_conflicts(out_lines), out_lines, exec_result
