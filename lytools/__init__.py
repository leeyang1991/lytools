# coding='utf-8'
__version__ = '0.0.65'

# from ._lytools import *
# from outdated import warn_if_outdated
from outdated import check_outdated
# warn_if_outdated("lytools", __version__)
# check_outdated("lytools", __version__)
is_outdated, latest = check_outdated("lytools", __version__)
if is_outdated:
    print("The package lytools is out of date. Your version is %s, the latest is %s." % (__version__, latest))