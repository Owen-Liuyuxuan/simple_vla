"""File I/O without mmcv."""
import os
import pickle


def mkdir_or_exist(path):
    if path:
        os.makedirs(path, exist_ok=True)


def dump(obj, path):
    mkdir_or_exist(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
