"""
Versions of key python libaries
"""
from __future__ import division, print_function
import sys
import numpy as np
import pandas as pd
import matplotlib


def show_versions():
    print('python    : %s.%s.%s' % sys.version_info[:3])
    print('numpy     : %s' % np.__version__)
    print('matplotlib: %s' % matplotlib.__version__)
    print('pandas    : %s '% pd.__version__)


print(__doc__)
show_versions()
print()
