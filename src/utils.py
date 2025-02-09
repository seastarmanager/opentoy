from functools import partialmethod

from tqdm import tqdm


def disable_tqdm():
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
