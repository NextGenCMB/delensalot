"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""

from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept

from delensalot.config.visitor import transform

from delensalot.core.opfilt.MAP import opfilt_iso_e, opfilt_iso_gpt, opfilt_iso_p, opfilt_iso_t, opfilt_p # these are iso MAP
from delensalot.core.opfilt.MAP import opfilt_t # these are aniso MAP with and without OBD


class iso_transformer:

    def build_opfilt_iso_pp(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'ffi': cf.ffi,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'wee': cf.k == 'p_p',
                'transf_b': cf.ttebl['b'],
                'nlev_b': cf.nlev_p,
            }
        return opfilt_iso_p.alm_filter_nlev_wl(**extract())
    
    def build_opfilt_iso_ee(self, cf):
        assert 0, "Implement if needed"
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'ffi': cf.ffi,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,   
            }
        return opfilt_iso_e.alm_filter_nlev_wl(**extract())
    
    def build_opfilt_iso_gmv(self, cf):
        assert 0, "Implement if needed"
        def extract():
            return {}
        return opfilt_iso_gpt.alm_filter_nlev_wl(**extract())

    def build_opfilt_iso_tt(self, cf):
        assert 0, "Implement if needed"
        def extract():
            return {}
        return opfilt_iso_t.alm_filter_nlev_wl(**extract())


class aniso_transformer:

    def build_opfilt_pp(self, cf):
        def extract():
            return {
                'ninv_geom': cf.ninvjob_geometry,
                'ninv': cf.ninv,
                'ffi': cf.ffi,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'sht_threads': cf.tr,
                'tpl': cf.tpl,
                'transf_blm': cf.ttebl['b'],
                'verbose': cf.verbose,
                'lmin_dotop': cf.min(cf.lmin_teb[1], cf.lmin_teb[2]),
                'wee': cf.k == 'p_p'
            }        
        return opfilt_p.alm_filter_ninv_wl(**extract())
    
    def build_opfilt_tt(self, cf):
        def extract():
            return {}
        return opfilt_t.alm_filter_ninv_wl(**extract())


@transform.case(DLENSALOT_Concept, iso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pbb']:
        return transformer.build_opfilt_iso_pp(expr)
    elif expr.k == 'pee':
        return transformer.build_opfilt_iso_ee(expr)
    elif expr.k == 'ptt':
        return transformer.build_opfilt_iso_tt(expr)
    elif expr.k == 'p':
        return transformer.build_opfilt_iso_gmv(expr)
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
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pbb']:
        return transformer.build_opfilt_iso_pp(expr)
    elif expr.k == 'pee':
        return transformer.build_opfilt_iso_ee(expr)
    elif expr.k == 'ptt':
        return transformer.build_opfilt_tt(expr)
    elif expr.k == 'p':
        assert 0, "implement if needed"
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