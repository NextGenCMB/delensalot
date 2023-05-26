"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept

from delensalot.config.visitor import transform

from delensalot.core.opfilt import bmodes_ninv, tmodes_ninv # these are merely for creating OBD (OTD) matrices
from delensalot.core.opfilt.QE import opfilt_iso_p, opfilt_iso_t, opfilt_p # these are iso QE
from delensalot.core.opfilt.QE import opfilt_t # these are aniso QE


class iso_transformer:

    def build_opfilt_iso_pp(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'transf': cf.ttebl['e'],
                'alm_info': cf.lm_max_unl,
                'wee': cf.k == 'p_p',
            }
        return opfilt_iso_p.alm_filter_nlev(**extract())

    def build_opfilt_iso_tt(self, cf):
        def extract():
            return {}
        return opfilt_iso_t.alm_filter_nlev(**extract())


class aniso_transformer:
    def build_opfilt_pp(self, cf):
        def extract():
            return {}
        return opfilt_p.alm_filter_ninv(**extract())
    
    def build_opfilt_tt(self, cf):
        def extract():
            return {}
        return opfilt_t.alm_filter_ninv(**extract())


@transform.case(DLENSALOT_Concept, iso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pee', 'pbb']:
        return transformer.build_opfilt_iso_pp(expr)
    elif expr.k in ['ptt']:
        return transformer.build_opfilt_iso_tt(expr)
    elif expr.k == 'p_te':
        assert 0, "implement if needed"
    elif expr.k == 'p_et':
        assert 0, "implement if needed"
    elif expr.k == 'pte':
        assert 0, "implement if needed"
    elif expr.k == 'p_tb':
        assert 0, "implement if needed"
    elif expr.k == 'pbt':
        assert 0, "implement if needed"
    elif expr.k == 'ptb':
        assert 0, "implement if needed"
    elif expr.k == 'pp':
        assert 0, "implement if needed"


@transform.case(DLENSALOT_Concept, aniso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pee', 'pbb']:
        return transformer.build_opfilt_pp(expr)
    elif expr.k in ['ptt']:
        return transformer.build_opfilt_tt(expr)
    elif expr.k == 'p_te':
        assert 0, "implement if needed"
    elif expr.k == 'p_et':
        assert 0, "implement if needed"
    elif expr.k == 'pte':
        assert 0, "implement if needed"
    elif expr.k == 'p_tb':
        assert 0, "implement if needed"
    elif expr.k == 'pbt':
        assert 0, "implement if needed"
    elif expr.k == 'ptb':
        assert 0, "implement if needed"
    elif expr.k == 'pp':
        assert 0, "implement if needed"