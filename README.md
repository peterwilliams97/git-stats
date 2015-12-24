# [git-stats](https://twitter.com/git_stats)
Compute and display statistics of git repositories

# Requirements
Scientific Python distribution such as Anaconda

e.g. Install Anaconda then
conda update conda
conda update anaconda

Tested with

    python    : 2.7.10
    numpy     : 1.10.1
    matplotlib: 1.5.0
    pandas    : 0.17.1

# [code-age](https://github.com/peterwilliams97/git-stats/blob/master/code-age.py)
Analyzes the age of files in a git repository and writes some reports and draws some graphs about them.


Writes reports in the directory structure given in [git.stats.tree.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats.tree.txt)

e.g. For repository [cpython](https://github.com/python/cpython.git)

    [root]                                    Defaults to ~/git.stats
      ├── cpython                             Directory for https://github.com/python/cpython.git
      │   └── reports
      │       ├── 2011-03-06.d68ed6fc.2_0     Revision `d68ed6fc` which was created on 2011-03-06 on
      │       │   │                           on branch `2.0`.
      │       │   └── __c.__cpp.__h           Report on *.c, *.cpp and *.h files in this revision
      │       │       ├── Guido_van_Rossum    Sub-report on author `Guido van Rossum`
      │       │       │   ├── code-age.png    Graph of code age. LoC / day vs date
      │       │       │   ├── code-age.txt    List of commits in the peaks in the code-age.png graph
      │       │       │   ├── details.csv     LoC in each directory in for these files and authors
      │       │       │   ├── newest.txt      List of newest commits for these files and authors
      │       │       │   └── oldest.txt      List of oldest commits for these files and authors


### A closer look at [git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/](https://github.com/peterwilliams97/git-stats/tree/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum)

#### [code-age.png](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.png)
![Age graph](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.png)


#### [code-age.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats/cpython/reports/2015-12-22.4120e146.2_7/__c.__cpp.__h/Guido_van_Rossum/code-age.txt)

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
