# TODO

## Danger Signs
* source file with many owners
* source file with many recent owners
* source file with much churn
* author changing many source files


## TODO
* separate versions.py file
* Top level stats
*   Histogram LoC / file
*   Histogram LoC / commit
*   Total # files
*   Author: Files contributed to
*   Author: LoC
*   Get spreadsheet working
* For each Report: csv of author, LoC, start, end,
* Add an extensions subdirectory level
* Add a BlameMap copy constructor + checks that members don't change. BlameMap needs to be immuatabe. Add accessor functions
* Save pickle as .old.pkl before changing it
* Trim ends of time series e.g 1% at each end, to remove outliers
* Reuse data between branches on same repository. commmon commits, diff then blame
* Make history.txt reflect peaks in history.png better
* path_list -> mask list. Test for file mask[s]
* Use http://www.pygit2.org/recipes/git-show.html
* Install like this: http://www.pygit2.org/install.html#libgit2-within-a-virtual-environment
* consistent naming. row, column. singular
* validate dates
* optimize across different versions. cache by commit
* Improve peak detection. Perhaps by scale
* If first bin is big then annotate it
* Test on python 3, windows
* Test on linux kernel reposistor
* Improve legend
* get_ => git_
* Force
* Douche-bags vs hipsters vs bogans (Bourgeois vs Proletariat vs Dilettante)
* Directory structure
*  top/ README, manifest
*   data/  Blame results
*   reports/
* Find java files that are web pages. e.g. matching .page, .html
* Better error messages
* Exclusions / Inclusions config
* Specify time zone
* Label peaks, 1,2,3, .. with table of short descriptions for each peak (biggest commits)
* Correlations. Developers, files. Sort by most correlated
* Principal components of LoC vs date. Look for clusters in PC[i] vs PC[j]
* Full table of developers x extensions
* List of files: full path, size (bytes), LoC, g|b|i
* Command line :
** authors (Peter | Tim)
** extensions (c | java)
** path patterns
* .gitstatsignore - using glob
* Plot age of code being replaced vs time
* Churn by module
* Source files by age. Mean stddev
* Warning signs: Code with many authors, recent code, big changes
* Authors who touch all modules !@#$
* tf/idf on commit messages
* Sort files by ownership most owner
* Ownership vector for each file {auther: frac loc} Do a tf/idf like classidicati
* Option to list/use saved data/ files
* !@#$TOMORROW: filter by extension(s), author(s)
* Enqueue / parallelize git commands
* Use a sequence number for mpl figure numbers
* dict of author aliases
* -@ options command line option. Also save options dict with each report
* trim trailing / from input directory name
* re-use subdirectories of previous blames
* add unit tests
* write blame ruuning time to file
* history: legend for eevryone to matche graph, graph individuals to match legend
* capitalize pandas consistently
* Interactive version. Using Bokeh https://www.dataquest.io/blog/python-data-visualization-libraries/
* multiprocessing https://docs.python.org/2/library/multiprocessing.html
* strip comments http://rosettacode.org/wiki/Strip_block_comments#Python
* unit tests. see what pandas does
* abspath in error message consistently
* avoid reblaming files that fail blame. keep a list of blamed files
* Tests: All modules, major file patterns, all authors
* Graph percentage of code by age. How?
* Get on http://github-awards.com/users?utf8=%E2%9C%93&type=city&language=Python&city=Melbourne
* Trim 0 LoC peaks
    TODO:
        hours of day
        days of week
        top times
        group code by functionality

directory structure
------------------
git.stats\
    directory {hash: description}
    hash:1\
        data\   git blame returns
        reports\
            authors,exts:1\
            authors,exts:2\
    hash:2\


# References
http://cse.unl.edu/~elbaum/papers/conferences/icsm98.pdf

