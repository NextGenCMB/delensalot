"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""

from delensalot.config.visitor import transform
from delensalot.core.opfilt import utils_cinv_p
from delensalot.core.opfilt import bmodes_ninv, tmodes_ninv # these are merely for creating OBD (OTD) matrices
from delensalot.core.opfilt import opfilt_iso_ee_wl, opfilt_iso_eenob_wl, opfilt_iso_gmv_wl, opfilt_iso_pp, opfilt_iso_tt, opfilt_iso_tt_wl
from delensalot.core.opfilt import opfilt_ee_wl, opfilt_ee_wl_dev, opfilt_pp, opfilt_tt, opfilt_tt_wl


class iso_transfomer:
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_ee_wl(**extract())
    
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_eenob_wl(**extract())
    
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_gmv_wl(**extract())
    
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_pp(**extract())
    
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_tt(**extract())
    
    def build_opfilt_ee_wl(cf):
        def extract():
            return {}
        return opfilt_iso_tt_wl(**extract())


class aniso_transformer:
    def build(cf):
        def extract():
            return {}
        return opfilt_ee_wl(**extract())
    
    def build(cf):
        def extract():
            return {}
        return opfilt_ee_wl_dev(**extract())
    
    def build(cf):
        def extract():
            return {}
        return opfilt_pp(**extract())
    
    def build(cf):
        def extract():
            return {}
        return opfilt_tt(**extract())
    
    def build(cf):
        def extract():
            return {}
        return opfilt_tt_wl(**extract())


@transform.case('p_p', iso_transfomer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_opfilt_ee_wl(expr)

@transform.case('p_eb', iso_transfomer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('p_be', iso_transfomer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('peb', iso_transfomer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('pee', iso_transfomer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('ptt', iso_transfomer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('pp', iso_transfomer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)


@transform.case('p_p', aniso_transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_opfilt_ee_wl(expr)

@transform.case('p_eb', aniso_transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('p_be', aniso_transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('peb', aniso_transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('pee', aniso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('ptt', aniso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case('pp', aniso_transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)