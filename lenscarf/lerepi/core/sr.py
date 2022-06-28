from logging import handlers
import os, sys

import logging
from logdecorator import log_on_start, log_on_end
log = logging.getLogger(__name__)

import numpy as np
import subprocess


# bashCommand = ["ls", "{}/p_p*/wflms/*".format(self.analysispath), "|", "wc" ,"-l"]
# process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, text=True, shell=True)
# output, error = process.communicate()
# log.info(output)


class analysisreport:
    def __init__(self, reportmodel):
        self.__dict__.update(reportmodel.__dict__)


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished")
    def collect_jobs(self):
        jobs = []
        jobs.append(0)
        self.jobs = jobs


    def count(self, filenames):
        wflm_c, wflm_mc = 0, 0
        btempl_c, btempl_mc = 0, 0
        if any("p_it{}.npy".format(self.itmax-1) in filename for filename in filenames):
            wflm_c += 1
        else:
            wflm_mc += 1
        if any("btempl_p{:03d}".format(self.itmax) in filename for filename in filenames):
            btempl_c += 1
        else:
            btempl_mc += 1

        return np.array([wflm_c, wflm_mc, btempl_c, btempl_mc])
    

    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):

        log.info("status report for {}".format(self.analysispath))
        for n in range(3):
            log.info("==============================================")
        for n in range(3):
            log.info("")
        wflm_c, wflm_mc = 0, 0
        btempl_c, btempl_mc = 0, 0
        counts = np.zeros(shape=4, dtype=np.int)
        for idx in self.jobs:
            for dirpath, dirnames, filenames in os.walk(self.analysispath):
                if dirpath.endswith('wflms'):
                    if self.version == '':
                        if len(dirpath.split('/')[-2]) == 11:
                            counts += self.count(filenames)
                    else:
                        if dirpath.split('/')[-2].endswith(self.version):
                            counts += self.count(filenames)

            log.info("{}/{} wflm{} (iteration {}) are already there".format(counts[0], np.sum(counts[0:2]), self.itmax-1, self.itmax))
            log.info("{}/{} btempl_p0{} are already there".format(counts[2], np.sum(counts[2:4]), self.itmax-1))
                
            
        for n in range(3):
            log.info("")
        for n in range(3):
            log.info("==============================================")
