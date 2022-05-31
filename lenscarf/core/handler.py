import os
from os.path import join as opj

import numpy as np

from plancklens.helpers import mpi

from lenscarf.iterators.statics import rec as Rec
from lenscarf.iterators import iteration_handler


class MAP_delensing():
    def __init__(self, survey_config, lensing_config, run_config):
        self.suvey_config = survey_config
        self.lensing_config = lensing_config
        self.run_config = run_config
        self.libdir_iterators = lambda qe_key, simidx, version: opj(run_config.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.ith = iteration_handler.transformer(lensing_config.iterator)


    def collect_jobs(self, libdir_iterators):
        jobs = []
        for idx in np.arange(self.run_config.imin, self.run_config.imax + 1):
            lib_dir_iterator = libdir_iterators(self.lensing_config.k, idx, self.run_config.v)
            if Rec.maxiterdone(lib_dir_iterator) < self.run_config.itmax:
                jobs.append(idx)
        self.jobs = jobs


    def run(self):
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.run_config.k, idx, self.run_config.v)
            if self.run_config.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < self.run_config.itmax:
                itlib = self.ith(self.run_config.k, idx, self.run_config.v, self.libdir_iterators, self.mf0, self.plm0, self.lensing_config, self.survey_config)
                itlib_iterator = itlib.get_iterator(idx)
                for i in range(self.run_config.itmax + 1):
                    # print("Rank {} with size {} is starting iteration {}".format(mpi.rank, mpi.size, i))
                    print("****Iterator: setting cg-tol to %.4e ****"%self.run_config.tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%self.run_config.soltn_cond(i))
                    itlib_iterator.chain_descr  = self.lensing_config.chain_descr(self.lens_config.lmax_unl, self.run_config.tol_iter(i))
                    itlib_iterator.soltn_cond   = self.lensing_config.soltn_cond(i)
                    print("doing iter " + str(i))
                    itlib_iterator.iterate(i, 'p')


class QE_delensing():
    def __init__(self, survey_config, lensing_config, run_config):
        assert 0, "Implement if needed"
        pass