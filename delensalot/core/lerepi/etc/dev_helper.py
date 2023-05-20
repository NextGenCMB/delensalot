"""dev_helper.py: somewhat hide hash-purging subroutine from users,
    as this should really only be used from developers.

    Reason you'd like to use this is, for example if you change the dirstructure of an existing analysis and don't want to rerun it,
    or you feed in the same noisemodel but from a different directory, to an existing analysis..
"""


import os, sys
from os.path import join as opj
import logging
from logdecorator import log_on_start, log_on_end
log = logging.getLogger(__name__)

from delensalot.core import mpi

dev_subr = "purgehashs"


def dev(parser, TEMP):
    if parser.purgehashs and mpi.rank == 0:
        def is_anadir(TEMP):
            if TEMP.startswith(os.environ['SCRATCH']):
                return True
            else:
                log.error('Not a $SCRATCH dir.')
                sys.exit()

        def get_hashfiles(TEMP):
            counter = 0
            hashfiles = []
            for dirpath, dirnames, filenames in os.walk(TEMP):
                _hshfile = [filename for filename in filenames if filename.endswith('hash.pk')]
                counter += len(_hshfile)
                if _hshfile != []:
                    hashfiles.append([dirpath, _hshfile])

            return hashfiles, counter

        if is_anadir(TEMP):
            log.info("====================================================")
            log.info("========        PURGING subroutine        ==========")
            log.info("====================================================")
            log.info("Will check {} for hash files: ".format(TEMP))
            hashfiles, counter = get_hashfiles(TEMP)
            if len(hashfiles)>0:
                log.info("I find {} hash files,".format(counter))
                log.info(hashfiles)
                userinput = input('Please confirm purging with YES: ')
                if userinput == "YES":
                    for pths in hashfiles:
                        for pth in pths[1]:
                            fn = opj(pths[0],pth)
                            os.remove(fn)
                            print("Deleted {}".format(fn))
                    print('All hashfiles have been deleted.')
                    hashfiles, counter = get_hashfiles(TEMP)
                    log.info("I find {} hash files".format(counter))  
                else:
                    log.info("Not sure what that answer was.")
            else:
                log.info("Cannot find any hash files.".format(counter))  

    log.info("====================================================")
    log.info("========        PURGING subroutine        ==========")
    log.info("====================================================")
