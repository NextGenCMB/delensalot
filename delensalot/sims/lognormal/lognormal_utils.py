"""Module with utilities for the lognormal sims module.
"""

import numpy as np
import flt
import healpy as hp


def get_out_quantities_from_a_map(outmap: np.ndarray) -> tuple:
    """
    Returns the mean, variance, skewness, lambda, muG and sigmaG from a given map.
    """
    skewness = get_skew_from_map(outmap)
    variance = get_variance_from_map(outmap)
    mean = get_mean_from_map(outmap)
    lamb = get_lambda_from_skew(skewness, variance, mean)

    alpha = get_alpha(mean, lamb)
    sigmaG = get_sigma_gauss(alpha, variance)
    muG = get_mu_gauss(alpha, variance)
    return mean, variance, skewness, lamb, muG, sigmaG


def create_lognormal_single_map(inputcl: np.ndarray, nside: int, lmax_gen: int, mu: float = 0.0, lamb: float = 0.0):
    """
    Creates a lognormal map with a given power spectrum and skewness. These are the parameters of the lognormal distribution that are specified with mu and lamb.
    """


    alpha = get_alpha(mu, lamb)
    xisinput = cl2xi(inputcl)/alpha/alpha

    xigaussian = np.log(xisinput+1)
    clgaussian = xi2cl(xigaussian)

    vargauss = np.dot(np.arange(1, 2*len(clgaussian), 2), clgaussian)/(4*np.pi)

    lmax_gen = 2*nside-1 if lmax_gen is None else lmax_gen
    almgaussian = hp.synalm(clgaussian, lmax = lmax_gen) #GENERATE TO HIGH LMAX
    maps = hp.alm2map(almgaussian, nside = nside, pol = False)
    
    #vargauss = np.array([xigaussian[i, i][0] for i in range(Nfields)])
    #vargauss = np.array([np.var(m) for m in maps])

    expmu = (mu+lamb)*np.exp(-vargauss*0.5)
    maps = np.array(maps)
    maps = np.exp(maps) 
    maps *= expmu
    maps -= lamb
    return maps


#shifted log-normal distribution
def shifted_lognormal_zero_mean(x, sigmaG, lamb):
    #equation (22) of https://www.aanda.org/articles/aa/pdf/2011/12/aa17294-11.pdf
    return np.exp(-(np.log(x/lamb+1)+sigmaG**2/2)**2./(2.*sigmaG**2.))/(x+lamb)/sigmaG/np.sqrt(2.*np.pi)*(x>-lamb)



def cl2xi(cl: np.ndarray, closed = False):
    """
    This goes from angular power spectrum to a correlation function calculated at theta points.
    """
    ls = np.arange(0, len(cl))
    factorcl = (2*ls+1)/(4*np.pi)
    coeffs = cl*factorcl
    return flt.idlt(coeffs, closed = closed)


def theta(n, closed = False):
    """
    Returns the theta for which the cl2xi are calculated, for a given n
    """
    return flt.theta(n, closed = closed)


def xi2cl(xi: np.ndarray, closed = False):
    """
    This goes from correlation function calculated at theta points to an angular power spectrum.
    """
    ls = np.arange(0, len(xi))
    factorcl = (2*ls+1)/(4*np.pi)
    return flt.dlt(xi, closed = closed)/factorcl



def get_mean_from_map(mappa: np.ndarray):
    """
    Mean from a map.
    """
    return np.mean(mappa)

def get_variance_from_map(mappa: np.ndarray):
    """
    Variance from a map.
    """
    return np.mean(mappa**2.)-np.mean(mappa)**2.

def get_skew_from_map(mappa: np.ndarray):
    """
    Skewness from a map.
    """
    return np.mean((mappa-get_mean_from_map(mappa))**3.)/np.mean(mappa**2.)**1.5

def y_skew(skew):
    """
    Formula (12) from https://arxiv.org/pdf/1602.08503.pdf
    It relates the skewnees to some factor that is used to get the lambda parameter for the log-normal generation.
    """
    result = 2+skew**2.+skew*np.sqrt(4+skew**2.)
    result /= 2
    return np.power(result, 1/3.)

def get_lambda_from_skew(skew, var, mu):
    lmbda = np.sqrt(var)/skew*(1+y_skew(skew)+1/y_skew(skew))-mu
    return lmbda 

def get_alpha(mu, lmbda):
    """
    Below formula (7) from https://arxiv.org/pdf/1602.08503.pdf
    """
    return mu+lmbda

def get_mu_gauss(alpha, var):
    """
    Gets the mu parameter for the Gaussian distribution for the log-normal
    """
    result = np.log(alpha**2./np.sqrt(var+alpha**2.))
    return result

def get_sigma_gauss(alpha, var):
    """
    Gets the sigma parameter for the Gaussian distribution for the log-normal.

    Here the variance is the variance of the wanted log-normal field.
    """
    result = np.log(1+var/alpha**2.)
    result = np.sqrt(result)
    return result


def suppress(l: np.ndarray, lsup: float = 7000, supindex: float = 10):
    """
    Suppression factor at high ell.
    """
    return np.exp(-1.0*np.power(l/lsup, supindex))

def suppress_cls(inputcl: np.ndarray, l: np.ndarray, lsup: float = 7000, supindex: float = 10):
    """
    Suppresses an input angular power spectrum at high ell.
    """
    return inputcl*suppress(l, lsup, supindex)

def process_cl(inputcl: np.ndarray, function: callable = suppress_cls, **kwargs):
    """
    Processes an input angular power spectrum with a given function.

    The function must take inputcl and l as arguments.
    
    Args:
        inputcl (np.ndarray): Input angular power spectrum.
        function (callable) default suppress_cls: Function to process the angular power spectrum.
        **kwargs: Keyword arguments for the function.
    
    Returns:
        np.ndarray: Processed angular power spectrum.
    """
    ls = np.arange(0, len(inputcl))
    return function(inputcl = inputcl, l = ls, **kwargs)
    