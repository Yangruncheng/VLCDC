from __future__ import absolute_import

from .make_model_clipreid import make_model_sg

__factory = {
    "sg": make_model_sg,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
