"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""

from delensalot.config.visitor import transform
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept
from delensalot.core.opfilt import QE_opfilt_iso_p, QE_opfilt_iso_t # these are iso QE
from delensalot.core.opfilt import QE_opfilt_aniso_p, QE_opfilt_aniso_t # these are aniso QE

from delensalot.core.opfilt import MAP_opfilt_iso_e, MAP_opfilt_iso_gpt, MAP_opfilt_iso_p, MAP_opfilt_iso_t # these are iso MAP
from delensalot.core.opfilt import MAP_opfilt_aniso_p, MAP_opfilt_aniso_t # these are aniso MAP with and without OBD


class QE_transformer:
    def build_iso(self, cf):
        return QE_iso_transformer
    def build_aniso(self, cf):
        return QE_aniso_transformer


class QE_iso_transformer:

    def build_opfilt_iso_pp(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'transf': cf.ttebl['e'],
                'alm_info': cf.lm_max_unl,
                'wee': cf.k == 'p_p',
            }
        return QE_opfilt_iso_p.alm_filter_nlev(**extract())

    def build_opfilt_iso_tt(self, cf):
        def extract():
            return {}
        return QE_opfilt_iso_t.alm_filter_nlev(**extract())


class QE_aniso_transformer:
    def build_opfilt_pp(self, cf):
        def extract():
            return {}
        return QE_opfilt_aniso_p.alm_filter_ninv(**extract())
    
    def build_opfilt_tt(self, cf):
        def extract():
            return {}
        return QE_opfilt_aniso_t.alm_filter_ninv(**extract())


class MAP_transformer:
    def build_iso(self, cf):
        return MAP_iso_transformer
    def build_aniso(self, cf):
        return MAP_aniso_transformer


class MAP_iso_transformer:

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
        return MAP_opfilt_iso_p.alm_filter_nlev_wl(**extract())
    
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
        return MAP_opfilt_iso_e.alm_filter_nlev_wl(**extract())
    
    def build_opfilt_iso_gmv(self, cf):
        assert 0, "Implement if needed"
        def extract():
            return {}
        return MAP_opfilt_iso_gpt.alm_filter_nlev_wl(**extract())

    def build_opfilt_iso_tt(self, cf):
        assert 0, "Implement if needed"
        def extract():
            return {}
        return MAP_opfilt_iso_t.alm_filter_nlev_wl(**extract())


class MAP_aniso_transformer:

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
        return MAP_opfilt_aniso_p.alm_filter_ninv_wl(**extract())
    
    def build_opfilt_tt(self, cf):
        def extract():
            return {}
        return MAP_opfilt_aniso_t.alm_filter_ninv_wl(**extract())


