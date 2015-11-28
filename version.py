"""
    Versions of key python libraries used by git-stats
"""
from __future__ import division, print_function
import sys
import numpy as np
import pandas as pd
import matplotlib

print(__doc__)
print('    python    : %s.%s.%s' % sys.version_info[:3])
print('    numpy     : %s' % np.__version__)
print('    matplotlib: %s' % matplotlib.__version__)
print('    pandas    : %s '% pd.__version__)
print()
