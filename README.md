# [git-stats](https://twitter.com/git_stats)
Compute and display statistics of git repositories

# Requirements
A scientific Python distribution such as Anaconda.

e.g. Install Anaconda then
    conda update conda
    conda update anaconda

Tested with

    python    : 2.7.10 and 3.5.1
    numpy     : 1.10.1
    matplotlib: 1.5.0
    pandas    : 0.17.1

You can check your versions with [version.py](https://github.com/peterwilliams97/git-stats/blob/master/version.py).

git-stats currently contains only one analysis script, [code-age.py](https://github.com/peterwilliams97/git-stats/blob/master/code-age.py).

# [code-age.py](https://github.com/peterwilliams97/git-stats/blob/master/code-age.py)

### usage

* Copy code-age.py to your computer
* Open a shell and cd to the root of the git repository you want to report
* `python code-age.py` NOTE: This can take hours to run a big repository as it blames every file in the repository.
* The location of the reports directory will be written to stdout
* Optionally try some patterns e.g. `python code-age.py '*.py'`, `python code-age.py docs`

code-age.py analyzes the age of files in a git repository and writes some reports and draws some graphs about them. It writes reports in the directory structure given in [git.stats.tree.txt](https://github.com/peterwilliams97/git-stats/blob/master/examples/git.stats.tree.txt)

NOTE: __LoC__ is short for Lines of Code.

e.g. For repository [git](https://github.com/python/cpython.git), which is a github mirror off the git source code:

    [root]                                    Defaults to ~/git.stats
      └── git                                 Directory for https://github.com/git/git.git
          └── reports
              └── 2015-12-29.28274d02.master  Revision 28274d02 which was created on 2015-12-22 on
                  │                           on branch "master".
                  └── [all-files]             Report on all files in this revision
                      ├── README              Summary of files in [all-files]
                      ├── author_ext_files.csv Number of files of given extension in which author has code
                      ├── author_ext_loc.csv  Number of LoC author in files of given extension by author
                      ├── [all-authors]       Sub-report on all authors
                      │   ├── README          Summary of files in [all-authors]
                      │   ├── code-age.png    Graph of code age. LoC / day vs date
                      │   ├── code-age.txt    List of commits in the peaks in the code-age.png graph
                      │   ├── details.csv     LoC in each directory in for these files and authors
                      │   ├── newest-commits.txt List of newest commits for these files and authors
                      │   └── oldest-commits.txt List of oldeswest commits for these files and authors
                ....
                      ├── Alex_Henrie         Sub-report on author Alex Henrie
                      │   ├── README
                      │   ├── code-age.png
                      │   ├── code-age.txt
                      │   ├── details.csv
                      │   ├── newest.txt
                      │   └── oldest.txt


### Top level files [2015-12-29.28274d02.master/\[all-files\]](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/)

##### 1) [README](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/READ)

This file contains summary information about this report.

    Summary of File Counts and LoC by Author and File Extension
    ===========================================================

    Totals: 2806 files, 23743 commits, 764802 LoC

    Revision Details
    ----------------
    Repository:  git (https://github.com/git/git.git)
    Date:        2015-12-29
    Description: master
    SHA-1 hash   28274d02c489f4c7e68153056e9061a46f62d7a0


##### 2) [author_ext_files.csv](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/author_ext_files.csv)

This shows the number of files in which each author has one or more lines of code in the revision by extension and author.
being reported. (This table shown on this page is truncated. [author_ext_files.csv](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/author_ext_files.csv) has the full table.)

NOTE: The numbers of files in the Total row and column are __not__ the number of files in the repository. They are the total numbers of files in which each author has one or more lines of
code.

<table><tr><th></th><th>Total</th><th>.c</th><th>.sh</th><th>.txt</th><th>.h</th></tr><tr><th>Total</th><th>40742.0</th><th>14352.0</th><th>9457.0</th><th>7276.0</th><th>2710.0</th></tr><tr><th>Junio C Hamano</th><th>7747.0</th><th>2779</th><th>1713</th><th>1973</th><th>504</th></tr><tr><th>Jeff King</th><th>3070.0</th><th>1479</th><th>853</th><th>296</th><th>291</th></tr><tr><th>Nguyễn Thái Ngọc Duy</th><th>1680.0</th><th>994</th><th>226</th><th>174</th><th>210</th></tr><tr><th>Shawn O. Pearce</th><th>1254.0</th><th>384</th><th>352</th><th>99</th><th>55</th></tr><tr><th>Jonathan Nieder</th><th>1159.0</th><th>300</th><th>373</th><th>305</th><th>62</th></tr><tr><th>Linus Torvalds</th><th>1088.0</th><th>770</th><th>78</th><th>16</th><th>136</th></tr><tr><th>Johannes Schindelin</th><th>1022.0</th><th>458</th><th>298</th><th>103</th><th>92</th></tr><tr><th>René Scharfe</th><th>761.0</th><th>514</th><th>104</th><th>60</th><th>73</th></tr><tr><th>Michael Haggerty</th><th>707.0</th><th>437</th><th>106</th><th>56</th><th>70</th></tr><tr><th>Thomas Rast</th><th>696.0</th><th>171</th><th>151</th><th>193</th><th>34</th></tr></table>

This shows the number of files in which each author has one or more lines of code in the revision
being reported. (This table shown on this page is truncated. [author_ext_files.csv](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/author_ext_files.csv) has the full table.)

##### 3) [author_ext_loc.csv](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/author_ext_loc.csv)


This shows the lines of code in the revision being reported by extension and author. (The table shown on this page is
truncated. author_ext_loc.csv](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/author_ext_loc.csv) has the full table.)

<table><tr><th></th><th>Total</th><th>.c</th><th>.sh</th><th>.po</th><th>.txt</th></tr><tr><th>Total</th><th>764802.0</th><th>198828.0</th><th>172727.0</th><th>159684.0</th><th>81591.0</th></tr><tr><th>Junio C Hamano</th><th>115080.0</th><th>37433</th><th>27753</th><th>6220</th><th>28929</th></tr><tr><th>Jeff King</th><th>31776.0</th><th>13134</th><th>11724</th><th>0</th><th>3175</th></tr><tr><th>Jiang Xin</th><th>24649.0</th><th>1170</th><th>718</th><th>11256</th><th>81</th></tr><tr><th>Shawn O. Pearce</th><th>24636.0</th><th>5392</th><th>4748</th><th>1519</th><th>2353</th></tr><tr><th>Nguyễn Thái Ngọc Duy</th><th>20908.0</th><th>13226</th><th>5499</th><th>0</th><th>1233</th></tr><tr><th>Peter Krefting</th><th>16243.0</th><th>4</th><th>11</th><th>15718</th><th>0</th></tr><tr><th>Alexander Shopov</th><th>16182.0</th><th>0</th><th>0</th><th>16149</th><th>29</th></tr><tr><th>Johannes Schindelin</th><th>15963.0</th><th>7531</th><th>4996</th><th>0</th><th>1345</th></tr><tr><th>Jonathan Nieder</th><th>15266.0</th><th>3111</th><th>6625</th><th>0</th><th>1914</th></tr><tr><th>Ævar Arnfjörð Bjarmason</th><th>14688.0</th><th>11093</th><th>1306</th><th>93</th><th>107</th></tr></table>



### A closer look at [2015-12-29.28274d02.master/\[all-files\]/\[all-authors\]](https://github.com/peterwilliams97/git-stats-examples/tree/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D)

This directory contains files that report on the age of all authors code for all files (i.e. every
file) in revision `28274d02`, the `git` repository `master` branch on 2015-12-29.

##### 1) [code-age.png](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/code-age.png) is a graph showing the age of the code in question.

The horizontal axis is date and the vertical axis is LoC /day. This means the area under the curve
between two dates is the LoC surviving from the period bounded by those datess.

You can see that some code from 2006 survives in the current git master branch.

![Age graph](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/code-age.png)


##### 2) [code-age.txt](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/code-age.txt) lists the commits in the peaks in code-age.png

    ================================================================================
    [all-authors]: 10 peaks 117654 LoC
    ................................................................................
      5) 187 commits 12253 LoC around 2007-07-18
     1025 LoC, 2007-07-21 90a7149 German translation for git-gui
     1006 LoC, 2007-07-22 e79bbfe Add po/git-gui.pot
      992 LoC, 2007-07-22 4fe7626 Italian translation of git-gui
     1095 LoC, 2007-07-25 2340a74 Japanese translation of git-gui
     1150 LoC, 2007-07-27 f6b7de2 Hungarian translation of git-gui
    ................................................................................
     10) 130 commits 9705 LoC around 2009-06-03
      135 LoC, 2009-05-26 3902985 t5500: Modernize test style
     7040 LoC, 2009-06-01 f0ed822 Add custom memory allocator to MinGW and MacOS builds
      124 LoC, 2009-06-04 195643f Add 'git svn reset' to unwind 'git svn fetch'
      127 LoC, 2009-06-06 2264dfa http*: add helper methods for fetching packs
      288 LoC, 2009-06-06 5424bc5 http*: add helper methods for fetching objects (loose)
    ................................................................................
        ...

##### 3) [oldest-commits.txt](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/oldest-commits.txt) lists the oldest commits in the code in question.

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

##### 4) [newest-commits.txt](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/newest-commits.txt) lists the oldest commits in the code in question.

    ================================================================================
    [all-authors]: 23743 commits 764802 LoC
    ................................................................................
       12 LoC, 2015-12-29 28274d0 Git 2.7-rc3
       11 LoC,   Documentation/RelNotes/2.7.0.txt
        1 LoC,   GIT-VERSION-GEN
    ................................................................................
      119 LoC, 2015-12-28 c5e5e68 l10n: Updated Bulgarian translation of git (2477t,0f,0u)
      119 LoC,   po/bg.po
    ................................................................................

       ...

##### 5) [details.csv](https://github.com/peterwilliams97/git-stats-examples/blob/master/examples/git.stats/git/reports/2015-12-29.28274d02.master/%5Ball-files%5D/%5Ball-authors%5D/details.csv) attempts to show where the code is distributed through the source tree.

<table><tr><th>dir</th><th>LoC</th><th>frac</th></tr><tr><th></th><th>764802</th><th>1</th></tr><tr><th>t</th><th>167955</th><th>0.21960585877128982</th></tr><tr><th>po</th><th>120787</th><th>0.15793237988394382</th></tr><tr><th>Documentation</th><th>82641</th><th>0.1080554182651196</th></tr><tr><th>builtin</th><th>55958</th><th>0.07316664966880317</th></tr><tr><th>git-gui</th><th>53934</th><th>0.0705202130747566</th></tr><tr><th>git-gui/po</th><th>37087</th><th>0.6876367412022101</th></tr><tr><th>contrib</th><th>35890</th><th>0.04692717853771303</th></tr><tr><th>gitk-git</th><th>29385</th><th>0.03842170914825013</th></tr><tr><th>compat</th><th>25884</th><th>0.03384405375508955</th></tr><tr><th>Documentation/RelNotes</th><th>19709</th><th>0.2384893696833291</th></tr></table>

