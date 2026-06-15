from __future__ import absolute_import

from .market1501 import Market1501
from .msmt17 import MSMT17

__factory = {
    "market1501": Market1501,
    "msmt17": MSMT17,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
