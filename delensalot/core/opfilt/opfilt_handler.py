"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""

from delensalot.core.opfilt import MAP_opfilt_iso_tp, QE_opfilt_iso_p, QE_opfilt_iso_t # these are iso QE
from delensalot.core.opfilt import QE_opfilt_aniso_p, QE_opfilt_aniso_t # these are aniso QE

from delensalot.core.opfilt import MAP_opfilt_iso_e, MAP_opfilt_iso_p, MAP_opfilt_iso_t # these are iso MAP
from delensalot.core.opfilt import MAP_opfilt_aniso_p, MAP_opfilt_aniso_t # these are aniso MAP with and without OBD


class QE_transformer:
    def build_iso(self, cf):
        return QE_iso_transformer
    def build_aniso(self, cf):
        return QE_aniso_transformer


class QE_iso_transformer:

    def build_opfilt_iso_p(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'transf': cf.ttebl['e'],
                'alm_info': cf.lm_max_unl,
                'wee': cf.k == 'p_p',
            }
        return QE_opfilt_iso_p.alm_filter_nlev(**extract())

    def build_opfilt_iso_t(self, cf):
        def extract():
            return {
                'nlev_t': cf.nlev_t,
                'transf': cf.ttebl['t'],
                'alm_info': cf.lm_max_unl,
            }
        return QE_opfilt_iso_t.alm_filter_nlev(**extract())


class QE_aniso_transformer:
    def build_opfilt_aniso_p(self, cf):
        def extract():
            return {
                'ninv_geom': cf.nivjob_geomlib,
                'ninv': cf.nivp_desc,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'sht_threads': cf.tr,
            }
        return QE_opfilt_aniso_p.alm_filter_ninv(**extract())
    
    def build_opfilt_aniso_t(self, cf):
        def extract():
            return {
                'ninv_geom': cf.nivjob_geomlib,
                'ninv': cf.nivt_desc,
                'transf': cf.ttebl['t'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'sht_threads': cf.tr,
            }
        return QE_opfilt_aniso_t.alm_filter_ninv(**extract())


class MAP_transformer:
    def build_iso(self, cf):
        return MAP_iso_transformer
    def build_aniso(self, cf):
        return MAP_aniso_transformer


class MAP_iso_transformer:

    def build_opfilt_iso_t(self, cf):
        def extract():
            return {
                'nlev_t': cf.nlev_t,
                'ffi': cf.ffi,
                'transf': cf.ttebl['t'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,   
            }
        return MAP_opfilt_iso_t.alm_filter_nlev_wl(**extract())
    
    def build_opfilt_iso_p(self, cf):
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
    
    def build_opfilt_iso_e(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'ffi': cf.ffi,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,   
            }
        return MAP_opfilt_iso_e.alm_filter_nlev_wl(**extract())
    
    def build_opfilt_iso_tp(self, cf):
        def extract():
            return {
                'nlev_p': cf.nlev_p,
                'nlev_t': cf.nlev_t,
                'ffi': cf.ffi,
                'transf': cf.ttebl['t'],
                'transf_e': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,   
            }
        return MAP_opfilt_iso_tp.alm_filter_nlev_wl(**extract())


class MAP_aniso_transformer:

    def build_opfilt_aniso_t(self, cf):
        def extract():
            return {
                'ninv_geom': cf.nivjob_geomlib,
                'ninv': cf.niv,
                'ffi': cf.ffi,
                'transf': cf.ttebl['t'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'sht_threads': cf.tr,
                'tpl': cf.tpl,
                'lmin_dotop': cf.lmin_teb[0],
                'verbose': cf.verbose,
            }    
        return MAP_opfilt_aniso_t.alm_filter_ninv_wl(**extract())
    
    def build_opfilt_aniso_p(self, cf):
        def extract():
            return {
                'ninv_geom': cf.nivjob_geomlib,
                'ninv': cf.niv,
                'ffi': cf.ffi,
                'transf': cf.ttebl['e'],
                'unlalm_info': cf.lm_max_unl,
                'lenalm_info': cf.lm_max_ivf,
                'sht_threads': cf.tr,
                'tpl': cf.tpl,
                'transf_blm': cf.ttebl['b'],
                'verbose': cf.verbose,
                'lmin_dotop': min(cf.lmin_teb[1], cf.lmin_teb[2]),
                'wee': cf.k == 'p_p'
            }        
        return MAP_opfilt_aniso_p.alm_filter_ninv_wl(**extract())
    
