from logging import handlers
import os, sys

from os.path import join as opj

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


    def count(self, filenames, it):
        wflm_c= 0
        btempl_c = 0

        if any("p_it{}.npy".format(it) in filename for filename in filenames):
            wflm_c += 1
        if any("btempl_p{:03d}".format(it) in filename for filename in filenames):
            btempl_c += 1

        return np.array([wflm_c, btempl_c])
    

    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):

        log.info("status report for {}".format(self.analysispath))
        for n in range(3):
            log.info("==============================================")
        for n in range(3):
            log.info("")

        qlms_dd_ct = 0
        for idx in self.jobs:
            for dirpath, dirnames, filenames in os.walk(self.analysispath):
                for fn in filenames:
                    with open(opj(dirpath,fn), 'r', encoding = 'latin1') as f:
                        first_line = f.readline(0)
                if dirpath.endswith('qlms_dd'):
                    qlms_dd_ct += len([filename for filename in filenames if filename.startswith("sim_p_p")])
        log.info("qlms:")
        log.info('------------------------')
        log.info("{}/{} QE phis are there".format(qlms_dd_ct, self.imax+1))
        
        counts = np.zeros(shape=2, dtype=np.int)
        for idx in self.jobs:
            log.info("")
            log.info("Wflms and B-templates:")
            log.info('------------------------')
            for it in np.arange(0,self.itmax+1):
                counts = np.zeros(shape=2, dtype=np.int)
                for dirpath, dirnames, filenames in os.walk(self.analysispath):
                    if dirpath.endswith('wflms'):
                        if self.version == '':
                            if len(dirpath.split('/')[-2]) == 11:
                                counts += self.count(filenames, it)
                        else:
                            if dirpath.split('/')[-2].endswith(self.version):
                                counts += self.count(filenames, it)
                log.info("it {}:".format(it))
                log.info("wflm{}: {}/{} ".format(it, counts[0], self.imax+1))
                log.info("btempl_p0{}: {}/{}".format(it, counts[1], self.imax+1))


        for n in range(3):
            log.info("")
        for n in range(3):
            log.info("==============================================")
