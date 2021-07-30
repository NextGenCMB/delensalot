import os
import numpy as np
from lenscarf.iterators import cs_iterator
from os.path import join as opj
import time


class logger(object):
    def __init__(self):
        pass

    def startup(self, iterator:cs_iterator.pol_iterator):
        """loger operations at startup """
        assert 0, 'implement this'

    def on_iterstart(self, itr:int, key:str, iterator:cs_iterator.pol_iterator):
        assert 0, 'implement this'

    def on_iterdone(self, itr:int, key:str, iterator:cs_iterator.pol_iterator):
        assert 0, 'implement this'

class logger_norms(logger):
    def __init__(self, txt_file):
        super().__init__()
        self.txt_file = txt_file
        self.ti = None

    def startup(self, iterator:cs_iterator.pol_iterator):
        if not os.path.exists(self.txt_file):
            with open(self.txt_file, 'w') as f:
                f.write('# Iteration step \n' +
                           '# Exec. time in sec.\n' +
                           '# Increment norm (normalized to starting point displacement norm) \n' +
                           '# Total gradient norm  (all grad. norms normalized to initial total gradient norm)\n' +
                           '# Quad. gradient norm\n' +
                           '# Det. gradient norm\n' +
                           '# Pri. gradient norm\n')
                f.close()

    def on_iterstart(self, itr:int, key:str, iterator:cs_iterator.pol_iterator):
        self.ti = time.time()

    def on_iterdone(self, itr:int, key:str, iterator:cs_iterator.pol_iterator):
        incr = iterator.hess_cacher.load('rlm_sn_%s_%s' % (itr-1, key))
        norm_inc = iterator.calc_norm(incr) / iterator.calc_norm(iterator.get_hlm(0, key))
        norms = [iterator.calc_norm(iterator.load_gradquad(itr - 1, key))]
        norms.append(iterator.calc_norm(iterator.load_graddet(itr - 1, key)))
        norms.append(iterator.calc_norm(iterator.load_gradpri(itr - 1, key)))
        norm_grad = iterator.calc_norm(iterator.load_gradient(itr - 1, key))
        norm_grad_0 = iterator.calc_norm(iterator.load_gradient(0, key))
        for i in [0, 1, 2]: norms[i] = norms[i] / norm_grad_0

        with open(opj(iterator.lib_dir, 'history_increment.txt'), 'a') as file:
            file.write('%03d %.1f %.6f %.6f %.6f %.6f %.6f \n'
                       % (itr, time.time() - self.ti, norm_inc, norm_grad / norm_grad_0, norms[0], norms[1], norms[2]))
            file.close()
