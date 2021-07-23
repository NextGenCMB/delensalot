import os
import numpy as np

class cacher(object):
    def cache(self, fn, obj):
        assert 0
    def load(self, fn):
        assert 0
    def is_cached(self, fn):
        assert 0
    def remove(self, fn):
        assert 0

class cacher_none(cacher):
    def cache(self, fn ,obj):
        pass
    def load(self, fn):
        assert 0
    def is_cached(self, fn):
        return False
    def remove(self, fn):
        pass

class cacher_npy(cacher):
    def __init__(self, lib_dir, verbose=False):
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        self.lib_dir = lib_dir
        self.verbose = verbose

    def _path(self, fn):
        return os.path.join(self.lib_dir, fn + '.npy')

    def cache(self, fn, obj):
        assert '.npy' not in fn
        np.save(os.path.join(self.lib_dir, fn + '.npy'), obj)
        if self.verbose: print("Cached " + fn + '.npy')

    def load(self, fn):
        assert '.npy' not in fn
        p = self._path(fn)
        assert os.path.exists(p), p
        if self.verbose:
            print("Loading " + fn + '.npy')
        return np.load(p)

    def is_cached(self, fn):
        return os.path.exists(self._path(fn))

    def remove(self, fn):
        assert self.is_cached(fn)
        os.remove(fn)


class cacher_mem(cacher):
    def __init__(self):
        self._cache = dict()

    def cache(self, fn, obj):
        self._cache[fn] = np.copy(obj)

    def load(self, fn):
        assert fn in self._cache.keys()
        return np.copy(self._cache[fn])

    def is_cached(self, fn):
        return fn in self._cache.keys()

    def remove(self, fn):
        assert fn in self._cache.keys()
        del self._cache[fn]