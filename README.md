# [git-stats](https://twitter.com/git_stats)
Compute and display statistics of git repositories

# Requirements
Scientific Python distribution such as Anaconda

e.g. Install Anaconda then
conda update conda
conda update anaconda

Tested with

    python    : 2.7.10 and 3.5.1
    numpy     : 1.10.1
    matplotlib: 1.5.0
    pandas    : 0.17.1

# [code-age](https://github.com/peterwilliams97/git-stats/blob/master/code-age.py)

Usage is [below](#code-agepy-usage).

Analyzes the age of files in a git repository and writes some reports and draws some graphs about them.

Writes reports in the directory structure given in [git.stats.tree.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats.tree.txt)

NOTE: __LoC__ is short for Lines of Code.

e.g. For repository [cpython](https://github.com/python/cpython.git)

    [root]                                    Defaults to ~/git.stats
      ├── cpython                             Directory for https://github.com/python/cpython.git
      │   └── reports
      │       ├── 2015-12-22.4120e146.2_7     Revision `4120e146` which was created on 2015-12-22 on
      │       │   │                           on branch `2.7`.
      │       │   └── __c.__cpp.__h           Report on *.c, *.cpp and *.h files in this revision
      │       │       ├── Guido_van_Rossum    Sub-report on author `Guido van Rossum`
      │       │       │   ├── code-age.png    Graph of code age. LoC / day vs date
      │       │       │   ├── code-age.txt    List of commits in the peaks in the code-age.png graph
      │       │       │   ├── details.csv     LoC in each directory in for these files and authors
      │       │       │   ├── newest.txt      List of newest commits for these files and authors
      │       │       │   └── oldest.txt      List of oldest commits for these files and authors


### A closer look at [2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/](https://github.com/peterwilliams97/git-stats/tree/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum)

This directory contains files that report on the age of `Guido van Rossum`'s `*.c`, `*.cpp` and `*.h`
code in revision `d68ed6fc`, the `cpython` `2.7` branch on 2015-12-22.

##### 1) [code-age.png](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.png) is a graph showing the age of the code in question.

The horizontal axis is date and the vertical axis is LoC /day. This means the area under the curve
between two dates is the LoC surviving from the period bounded by those datess.

You can see that some of Guido's C code from 1991 survives in the current Python 2.7 revision as
does code up to 2008.

![Age graph](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.png)


##### 2) [code-age.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.txt) lists the commits in the peaks in code-age.png

    ================================================================================
    Guido van Rossum: 10 peaks 12971 LoC
    ................................................................................
      2) 4 commits 941 LoC around 1991-08-10
      828 LoC, 1991-08-07 f133238 Initial revision
       31 LoC, 1991-08-08 a110409 Adde get_mouse and find_first/find_last (by robertl)
       76 LoC, 1991-08-08 f33866a Lots of cosmetic changes. Lots of small bugfixes (lint!). Made bgn_group and end_group form methods instead of top-level functions.
        6 LoC, 1991-08-08 9bec514 Fixed almost all list errors.

        ...
    ................................................................................
      1) 21 commits 4133 LoC around 1997-04-08
      111 LoC, 1997-04-02 783828f Added replace() implementation by Perry Stoll (debugged and reformatted by me).
     2540 LoC, 1997-04-04 333c03c New version by Sjoerd, with support for IRIX 6 audio library.
       23 LoC, 1997-04-10 dc02de7 Unknown changes by Jim Fulton.
     1065 LoC, 1997-04-10 7e5a815 Jim Fulton's version 2.2.
      295 LoC, 1997-04-12 db0be79 Completely revamped the way the default path is constructed.

        ...

##### 3) [oldest.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/oldest.txt) lists oldest commits in the code in question.

Guido's oldest C code commit that survives in the 2.7 branch is from October 1990. You can also see
that 2019 of his C commits survive, probably only partially). These remnants comprise 78,000
lines of C (i.e. `*.c`, `*.cpp` and `*.h`).

    ================================================================================
    Guido van Rossum: 2109 commits 78288 LoC
    ................................................................................
     3760 LoC, 1990-10-14 daadddf Initial revision
      802 LoC,   Modules/cstubs
      599 LoC,   Parser/pgen.c
      383 LoC,   Modules/cgen.py
    ................................................................................
       16 LoC, 1990-10-15 fddb56c Adde dconvenience functions.
       16 LoC,   Python/errors.c
    ................................................................................
        3 LoC, 1990-10-15 5122aee Made exception objects extern. Added convenience functions.
        3 LoC,   Include/pyerrors.h

        ...

##### 4) [newest.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/newest.txt) lists newest commits in the code in question.

Guido's last C code commit that survives in the 2.7 branch is from November 2008

    ================================================================================
    Guido van Rossum: 2109 commits 78288 LoC
    ................................................................................
        3 LoC, 2008-09-11 e9f5f91 - Issue #3629: Fix sre "bytecode" validator for an end case.   Reviewed by Amaury.
        4 LoC,   Lib/test/test_re.py
        3 LoC,   Modules/_sre.c
        2 LoC,   Misc/NEWS
    ................................................................................
       16 LoC, 2008-08-20 5512f4f Issue 1179: [CVE-2007-4965] Integer overflow in imageop module.
       16 LoC,   Modules/imageop.c

       ...

##### 5) [details.csv](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/details.csv) attempts to show where the code is distributed through the source tree.

<table>
<tr><th>dir</th><th>LoC</th><th>frac</th></tr>
<tr><td></td><td>78288</td><td>0</td></tr>
<tr><td>Modules</td><td>33059</td><td>0.422274167178</td></tr>
<tr><td>Objects</td><td>17963</td><td>0.22944768036</td></tr>
<tr><td>Python</td><td>13024</td><td>0.166360106274</td></tr>
<tr><td>Include</td><td>4256</td><td>0.0543633762518</td></tr>
<tr><td>PC</td><td>3583</td><td>0.045766911915</td></tr>
<tr><td>Parser</td><td>3248</td><td>0.0414878397711</td></tr>
</table>

* The source tree of Guido's C files contains 78288 LoC
* Its subdirectory `Modules` contains 33059 LoC
* etc


### Top level files [2015-12-22.4120e146.2_7/__c.__cpp.__h/](https://github.com/peterwilliams97/git-stats/tree/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/)

##### 1) [ext_author_files.csv](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/ext_author_files.csv) shows numbers of source files by extension and author

<table><tr><th></th><th>Total</th><th>Guido van Rossum</th><th>Martin v. Löwis</th><th>Tim Peters</th><th>Neal Norwitz</th></tr><tr><th>Total</th><th>15092</th><th>3252</th><th>1239</th><th>962</th><th>770</th></tr><tr><th>.c</th><th>12887</th><th>2744</th><th>1013</th><th>829</th><th>708</th></tr><tr><th>.h</th><th>2205</th><th>508</th><th>226</th><th>133</th><th>62</th></tr></table>

This shows the number of files in which each author has one or more lines of code in the revision
being reported. (This table shown on this page is truncated.)

##### 2) [ext_author_loc.csv](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/ext_author_loc.csv) shows LoC by extension and author

<table><tr><th></th><th>Total</th><th>Jack Jansen</th><th>Guido van Rossum</th><th>Martin v. Löwis</th><th>Thomas Heller</th></tr><tr><th>Total</th><th>549292</th><th>79897</th><th>78288</th><th>49728</th><th>25848</th></tr><tr><th>.c</th><th>464959</th><th>79578</th><th>70939</th><th>40661</th><th>22891</th></tr><tr><th>.h</th><th>84333</th><th>319</th><th>7349</th><th>9067</th><th>2957</th></tr></table>

This shows the lines of code in the revision being reported. (This table shown on this page is truncated.)

### code-age.py usage

* Copy code-age.py to your computer
* Open a shell and cd to the root of the git repository you want to report
* `python code-age.py` NOTE: This can take hours to run a big repository as it blames every file in the repository.
* The location of the reports directory will be written to stdout
* Optionally try some patterns e.g. `python code-age.py '*.py'`, `python code-age.py docs`
