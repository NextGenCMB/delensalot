import numpy as np
from os.path import join as opj

from plancklens.helpers import mpi

from lenscarf.iterators.statics import rec as Rec
from lenscarf.iterators import iteration_handler


class MAP_delensing():
    def __init__(self, survey_config, lensing_config, run_config):
        self.suvey_config = survey_config
        self.lensing_config = lensing_config
        self.run_config = run_config
        self.libdir_iterators = lambda qe_key, simidx, version: opj(run_config.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)


    def collect_jobs(self, libdir_iterators):
        jobs = []
        for idx in np.arange(self.run_config.imin, self.run_config.imax + 1):
            lib_dir_iterator = libdir_iterators(self.lensing_config.k, idx, self.run_config.v)
            if Rec.maxiterdone(lib_dir_iterator) < self.run_config.itmax:
                jobs.append(idx)
        self.jobs = jobs


    def get_qest():
        pass


    def get_iterator(self, k, idx, v, cg_tol):
        ith = iteration_handler.scarf_iterator_pertmf(k, idx, v, cg_tol, self.libdir_iterators, self.lensing_config)
        return ith.get_iterator(idx)


    def run(self):
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.run_config.k, idx, self.run_config.v)
            if self.run_config.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < self.run_config.itmax:
                itlib = self.get_iterator(self.lensing_config.k, idx, self.lensing_config.v, self.run_config.cg_tol)
                for i in range(self.run_config.itmax + 1):
                    # print("Rank {} with size {} is starting iteration {}".format(mpi.rank, mpi.size, i))
                    print("****Iterator: setting cg-tol to %.4e ****"%self.run_config.tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%self.run_config.soltn_cond(i))

                    itlib.chain_descr  = self.chain_descrs(self.lens_config.lmax_unl, self.run_config.tol_iter(i))
                    itlib.soltn_cond   = self.soltn_cond(i)
                    print("doing iter " + str(i))
                    itlib.iterate(i, 'p')


class QE_delensing():
    def __init__(self, survey_config, lensing_config, run_config):
        assert 0, "Implement if needed"
        pass