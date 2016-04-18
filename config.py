# -*- coding: utf-8 -*-
"""
    Wrapper around git command line interface

"""
from __future__ import division, print_function


#
# Configuration.
#
TIMEZONE = 'Australia/Melbourne'    # The timezone used for all commit times. TODO Make configurable
SHA_LEN = 8                         # The number of characters used when displaying git SHA-1 hashes
STRICT_CHECKING = True              # For validating code.
