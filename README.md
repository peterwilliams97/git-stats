# [gitstats](https://twitter.com/git_stats)
Compute and display statistics of git repositories

# Requirements
Scientific Python distribution such as Anaconda

e.g. Install Anaconda then
conda update conda
conda update anaconda

Tested with

    python    : 2.7.5
    numpy     : 1.9.2
    matplotlib: 1.4.3
    pandas    : 0.16.2

# [code-age](https://github.com/peterwilliams97/git-stats/blob/master/code-age.py)
Analyzes the age of files in a git repository and writes some reports and draws some graphs about them.


Writes reports in the following locations

e.g. For repository _linux_

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


* _[root]_ defaults to `~/git.stats`
* _linux_ is the remote name of the repository being analyzed. It was extracted from https://github.com/torvalds/linux.git
* _2015-11-25.6ffeba96_ contains reports on revision `6ffeba96`. `6ffeba96` was
        created on 2015-11-25. We hope that putting the date in the directory name makes it easier to navigate.
* \_\__sh_ is the directory containing reports on _*.sh_ files
* _Adrian_Hunter_ is the directory containing reports on author Adrian Hunter

A closer look at  `[root]/linux/reports/2015-11-25.6ffeba96/_sh/Adrian_Hunter`

## history[all].png
![Age graph](https://github.com/peterwilliams97/git-stats/blob/master/history%5Ball%5D.png)


## history[all].txt

    ================================================================================
    [all]: 22 peaks 237104 LoC
    ................................................................................
     20) 2 commits 36 LoC around 2002-11-03
       34 LoC, 2002-11-01 cf5c05a Added tests to stats.  Added nanXXXX functions.  Fixed kurtosis and skew to handle biased and unbiased estimates to match MATLAB.
        2 LoC, 2002-11-02 815de7e Changed takemask to extract.  Added inverse of extract called insert.  Extract takes 1-d array out of raveled nd array according to raveled mask.  Insert puts elements from 1-d array back into an n-d array according to mask.  It can be thought of as the inverse of extract.  It allows for operations to only be performed when necessary.
    ................................................................................
     22) 0 commits 0 LoC around 2002-12-31
    ................................................................................
     18) 7 commits 244 LoC around 2003-10-01
       32 LoC, 2003-09-21 ba4e597 Moved arraymap from cephes to scipy_base.  Renamed from general_function to vectorize
        9 LoC, 2003-09-23 2634919 Added test functions for extract and insert.
      189 LoC, 2003-10-02 56a1b11 Introduced machar - Machine Arithmetics - module. It resolves the inaccuracy problem in limits, in addition, machar computes many other constants of floating point arithmetic system. machar_double and machar_float instances are all you need to import from machar.
    ................................................................................
     19) 2 commits 105 LoC around 2004-11-26
       52 LoC, 2004-11-18 ca66987 Implemented mintypecode.
       53 LoC, 2004-12-04 5b13d29 numarray port "first draft" changes.
    ................................................................................
      4) 78 commits 21157 LoC around 2005-09-26
    17483 LoC, 2005-09-27 571c53f Added cvs version 1.46.23.2019 of f2py2e to svn
     1469 LoC, 2005-09-28 1e7839d  r3164@803638d6:  kern | 2005-09-26 01:27:54 -0700  Added scipy.lib.mtrand. It is not yet integrated into scipy.stats
      579 LoC, 2005-10-04 50967f0 Added cblas.h.
    ................................................................................
     16) 64 commits 1089 LoC around 2006-07-23
      272 LoC, 2006-07-26 9bda193 Add broadcasting behavior to random-number generators.  Fix cholesky to keep matrix return.
       71 LoC, 2006-07-29 b878328 Fix #114: Problems with building with MSVC and GCC under Cygwin
      138 LoC, 2006-07-31 162f816 Move docstrings from multiarraymodule.c to add_newdocs.py.
    ................................................................................
     14) 57 commits 3110 LoC around 2007-04-01
      346 LoC, 2007-03-31 e784b53 Add tests for clipping.  Also some good tests on choose and convert_tocommontype.  Fix some mixed-type problems.  Fix problems with clipping. More to be done to close ticket #425
      228 LoC, 2007-04-05 081af11 Split Series header/code/interface tests into Vector and Matrix components
      608 LoC, 2007-04-06 c6992ef Removed test1D.py, test2D.py and test3D.py in favor of testVector.py, testMatrix.py and testTensor.py because these new tests use inheritance to duplicate same tests for different data types
    ................................................................................
     10) 26 commits 3073 LoC around 2007-12-04
      800 LoC, 2007-11-28 6c00b11 use 'in' keyword to test dictionary membership
     1096 LoC, 2007-11-30 49a0503 * Added a new typemap suite, ARGOUTVIEW, which takes a view of a data   buffer and converts it to an output numpy array (1D, 2D and 3D, with   before- and after- dimension specifications.)
      653 LoC, 2007-11-30 b0e8c78 * Added support for FORTRAN-ordered arrays to numpy.i.
    ................................................................................
      2) 63 commits 14938 LoC around 2008-11-21
     1303 LoC, 2008-11-22 9ac837a Merge branch 'ufunc'
    12199 LoC, 2008-11-23 03582a3 Moved numpy-docs under doc/ in the main Numpy trunk.
      310 LoC, 2008-11-27 ce7cd10 Doc update
    ................................................................................
      9) 25 commits 2880 LoC around 2009-06-20
      330 LoC, 2009-06-11 910d3db commit 2e402e05f64912a3568a3e6351f1ffcf3fae601a Author: Robert Kern <robert.kern@gmail.com> Date:   Mon Jun 8 11:36:20 2009 -0500
      481 LoC, 2009-06-12 f2392bc Working on date-time...
     1461 LoC, 2009-06-20 87fa5ae Merge from doc wiki
    ................................................................................
     12) 88 commits 2885 LoC around 2009-10-28
      325 LoC, 2009-10-22 bb129ca * Added docs on i/o with focus on genfromtxt
      295 LoC, 2009-10-27 f676229 Add first cut of C code coverage tool
      196 LoC, 2009-10-30 ac72fa1 ENH: implement single and double precision nextafter* for npymath.
    ................................................................................
     11) 26 commits 2128 LoC around 2010-06-03
      129 LoC, 2010-05-26 6be6945 STY: Some c coding style cleanups.
       39 LoC, 2010-06-02 b42757e DOC: merge wiki edits for module linalg.
     1718 LoC, 2010-06-06 e520cdd DOC: add automatic documentation generation from C sources (using Doxygen at the moment)
    ................................................................................
      5) 14 commits 3453 LoC around 2010-09-23
       33 LoC, 2010-09-14 96afea0 * ma.core._print_templates: switched the keys 'short' and 'long' to 'short_std' and 'long_std' respectively (bug #1586) * Fixed incorrect broadcasting in ma.power (bug #1606)
       34 LoC, 2010-09-21 14d8e20 BUG: core: ensure cfloat and clongdouble scalars have a __complex__ method, so that complex(...) cast works properly (fixes #1617)
     3243 LoC, 2010-09-23 96c4eea ENH: First commit of hermite and laguerre polynomials. The documentation and tests still need fixes.
    ................................................................................
      8) 36 commits 2166 LoC around 2010-12-24
       88 LoC, 2010-12-15 b9f1c9a ENH: iter: Expose noinner support to the Python iterator interface
      113 LoC, 2010-12-16 e2d8b78 ENH: iter: Add ability to automatically determine output data types
     1022 LoC, 2010-12-21 6d41baf ENH: Implemented basic buffering
    ................................................................................
      3) 89 commits 10851 LoC around 2011-01-20
     2181 LoC, 2011-01-15 5245d39 ENH: iter: Add support for buffering arrays with fields and subarrays
      660 LoC, 2011-01-24 a41de3a ENH: core: Start einsum function, add copyright notices to files
      821 LoC, 2011-01-25 e81e8da ENH: core: Add SSE intrinsic support to some of the einsum core loops
    ................................................................................
      6) 95 commits 12011 LoC around 2011-07-02
     1884 LoC, 2011-06-28 b2ac4ad ENH: umath: Move the type resolution functions into their own file
     3073 LoC, 2011-07-08 36f4bdf ENH: nditer: Move construction/copy/destruction to its own implementation file
     2667 LoC, 2011-07-08 62a5ce1 ENH: nditer: Move the non-templated API into its own file
    ................................................................................
      7) 28 commits 6021 LoC around 2011-12-22
      961 LoC, 2011-12-21 00ff295 TST: Add tests for multidimensional coefficient array functionality.
     1318 LoC, 2011-12-28 175d90a DOC: Revise documentation for the basic functions.
      746 LoC, 2011-12-29 dc7719f DOC: Finish documenting new functions in the polynomial package.
    ................................................................................
      1) 38 commits 144360 LoC around 2012-10-19
      379 LoC, 2012-10-10 234523c working eig and eigvals priority 2 functions.
      365 LoC, 2012-10-12 4c9f286 svd implemented. Single output working. Multiple options not functional due to a bug in the harness.
    142659 LoC, 2012-10-18 c679f7b lapack_lite for builds of umath_linalg without an optimized lapack in the system.
    ................................................................................
     13) 79 commits 2913 LoC around 2013-10-08
      343 LoC, 2013-10-02 fd5d308 BUG: core: ensure __r*__ has precedence over __numpy_ufunc__
      341 LoC, 2013-10-05 9c7e6e3 BUG: Refactor nanfunctions to behave as agreed on for 1.9.
      470 LoC, 2013-10-13 9f1c178 Convert docstrings to comments for nose; PEP8 cleanup (some tests activated)
    ................................................................................
     15) 92 commits 3393 LoC around 2014-03-21
      298 LoC, 2014-03-13 9472a8d clean up in prep for python-ideas
      842 LoC, 2014-03-22 1eb81b7 ENH, MAINT: Use an abstract base class for the polynomial classes.
      350 LoC, 2014-03-22 a2c96a6 DOC: Fixup documentation for new way of generating classes.
    ................................................................................
     17) 8 commits 240 LoC around 2014-11-04
       28 LoC, 2014-10-26 473a386 DOC: add release notes for 1.9.1
       55 LoC, 2014-10-29 b40e686 BUG: fix not returning out array from ufuncs with subok=False set
      138 LoC, 2014-11-10 1bce8d7 BUG: Fix astype for structured array fields of different byte order.
    ................................................................................
     21) 3 commits 51 LoC around 2015-09-14
       23 LoC, 2015-09-20 f404885 DOC: update release notes for 1.9.3
        3 LoC, 2015-09-20 7f96e9d MAINT: set versions for 1.9.3 release
       25 LoC, 2015-09-20 a6d32de BLD: disable broken msvc14 trigonometric functions



## oldest[all].txt

    ================================================================================
    [all]: 7606 commits 522006 LoC
    ................................................................................
      296 LoC, 2002-03-30 0562713 Refactored to create scipy_base.
      272 LoC,   numpy/lib/polynomial.py
       23 LoC,   numpy/lib/scimath.py
        1 LoC,   numpy/core/__init__.py
    ................................................................................
        6 LoC, 2002-04-01 1af7875 added limits.py to scipy_base.  It looks like Travis O. began this process, but didn't quite get to checking it in.  I used the 1.13 version from the attic in scipy (the latest version there).  Hope that is the right one.
        6 LoC,   numpy/core/getlimits.py
    ................................................................................
        2 LoC, 2002-04-02 5dcb208 Finished special functions. (Added orthogonal polynomials).
        2 LoC,   numpy/lib/scimath.py
    ................................................................................
      276 LoC, 2002-04-03 f897e1f major reorg of scipy_base -- initial checkin
       92 LoC,   numpy/lib/index_tricks.py
       67 LoC,   numpy/lib/shape_base.py
       52 LoC,   numpy/lib/function_base.py
    ................................................................................
      349 LoC, 2002-04-03 44869d6 added test methods for scipy_base
      106 LoC,   numpy/lib/tests/test_shape_base.py
      100 LoC,   numpy/lib/tests/test_twodim_base.py
       74 LoC,   numpy/lib/tests/test_function_base.py

