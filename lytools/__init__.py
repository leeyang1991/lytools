# coding='utf-8'
__version__ = '0.0.80'

from ._lytools import *
from outdated import check_outdated
is_outdated, latest = check_outdated("lytools", __version__)
if is_outdated:
    print("The package lytools is out of date. Your version is %s, the latest is %s." % (__version__, latest))