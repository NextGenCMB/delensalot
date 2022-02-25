import matplotlib as mpl

def set_mpl():
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['font.size'] = 20
    # mpl.rcParams['figure.figsize'] = 6.4, 4.8
    mpl.rcParams['figure.figsize'] = 8.5, 5.5

    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rc('text', usetex=True)
    # mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mpl.rcParams['errorbar.capsize'] = 4
    mpl.rc('legend', fontsize=15)


def ptk(ls):
    return ls**2*(ls+1)**2/4