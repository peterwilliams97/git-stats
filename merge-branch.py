# -*- coding: utf-8 -*-
"""
    Run on a clean working copy.

    You can clean your current working copy with the following command:

    git reset --hard HEAD
"""
from __future__ import division, print_function
from git_calls import *


def report_failure(msg):
    """Each application will replace this with its own reporter
    """


def check_error(condition, msg):
    """If `condition` is True then report error message `msg` to `report_failure` and exit.
    """
    if condition:
        report_failure(msg)
    assert condition, msg


def merge_branches(whence, whither):
    print('=' * 80)
    print(__doc__)
    print('=' * 80)
    print('merge_branches: %s -> %s' % (whence, whither))

    ret, sha_before, err = git_show_sha()
    check_error(ret == 0, format_error('git_show_sha', ret, sha_before, err))

    ret, branch_list, err = git_track_all()
    check_error(whither in branch_list,
                'To branch "%s" not in branches %s' % (whither, sorted(branch_list)))
    check_error(whence in branch_list,
                'From branch "%s" not in branches %s' % (whence, sorted(branch_list)))

    ret, out, err = git_fetch()
    check_error(ret == 0, format_error('git_fetch', ret, out, err))

    ret, out, err = git_checkout(whither)
    check_error(ret == 0, format_error('git_checkout', ret, out, err))

    ret, out, err = git_pull(whither)
    check_error(ret == 0, format_error('git_pull', ret, out, err))

    ret, out, err = git_merge(whence)
    check_error(ret == 0, format_error('git_merge', ret, out, err))

    ret, sha_after, err = git_show_sha()
    check_error(ret == 0, format_error('git_show_sha', ret, sha_after, err))

    print(',' * 80)
    print('sha_before=%s' % sha_before)
    print('sha_after =%s' % sha_after)
    print('-' * 80)
    if sha_before == sha_after:
        print('No change')
    else:
        ret, info, err = git_show()
        print(info)
    print('=' * 80)

    # CONFLICT (content): Merge conflict in server/src/java/biz/papercut/pcng/ext/device/ExtDeviceConfigOption.java


def main():
    import optparse

    lowpriority()

    parser = optparse.OptionParser('python ' + sys.argv[0] + ' [options] [<directory>]')
    parser.add_option('-p', '--push', action='store_true', default=False, dest='do_push',
                      help='Push commit to origin')
    parser.add_option('-t', '--to', dest='whither', default='develop',
                      help='Branch to merge to')
    parser.add_option('-f', '--from', dest='whence', default='release/v15.3',
                      help='Branch to merge from')

    options, _ = parser.parse_args()

    merge_branches(options.whence, options.whither)

if __name__ == '__main__':

    main()
    print()
