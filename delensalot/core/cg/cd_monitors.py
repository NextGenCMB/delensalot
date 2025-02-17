import sys
import numpy as np
import time

## monitors
logger_basic = (
lambda iter, eps, watch=None, **kwargs: sys.stdout.write('[' + str(watch.elapsed()) + '] ' + str((iter, eps)) + '\n'))
logger_none = (lambda iter, eps, watch=None, **kwargs: 0)

class dt:
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                    np.floor(np.mod(self.dt, 60 * 60) / 60),
                                    np.floor(np.mod(self.dt, 60))))

    def __int__(self):
        return int(self.dt)
    

class stopwatch:
    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        lt = time.time()
        ret = (dt(lt - self.st), dt(lt - self.lt))
        self.lt = lt
        return ret

    def elapsed(self):
        lt = time.time()
        ret = dt(lt - self.st)
        self.lt = lt
        return ret
        
class monitor_basic:
    """Simple iterative solver monitor and convergence appraiser.

    This uses by default as convergence criterium the norm of the residual vector
    to decrease by a factor *eps_min* from the initial guess.
    The norm is calculated from the *dot_op* argument.

    """
    def __init__(self, dot_op, iter_max=1000, eps_min=1.0e-10, logger=logger_basic, d0=None):
        self.dot_op = dot_op
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.logger = logger
        self.d0 = d0

        self.watch = stopwatch()

    
    def criterion(self, iter, soltn, resid):
        delta = self.dot_op(resid, resid)

        if (iter == 0) and (self.d0 is None):
            self.d0 = delta

        if self.logger is not None: self.logger(iter, np.sqrt(delta / self.d0), watch=self.watch,
                                                  soltn=soltn, resid=resid)
        if (iter >= self.iter_max) or (delta <= self.eps_min ** 2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)
