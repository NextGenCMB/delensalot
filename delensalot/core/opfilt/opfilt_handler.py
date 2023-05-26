"""opfilt.opfilt_handler.py: interface to opfilt via transformer
"""

from delensalot.config.visitor import transform
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept

from delensalot.core.opfilt.QE.opfilt_handler import iso_transformer as iso_QE, aniso_transformer as aniso_QE # these are QE
from delensalot.core.opfilt.MAP.opfilt_handler import iso_transformer as iso_MAP, aniso_transformer as aniso_MAP # these are MAP


class QE_transformer:
    def build_iso(self, cf):
        return iso_QE
    def build_aniso(self, cf):
        return aniso_QE

class MAP_transformer:
    def build_iso(self, cf):
        return iso_MAP
    def build_aniso(self, cf):
        return aniso_MAP


@transform.case(DLENSALOT_Concept, QE_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.qe_filter_directional in ['isotropic']:
        return transformer.build_iso(expr)
    elif expr.qe_filter_directional in ['anisotropic']:
        return transformer.build_aniso(expr)
    

@transform.case(DLENSALOT_Concept, MAP_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.it_filter_directional in ['isotropic']:
        return transformer.build_iso(expr)
    elif expr.it_filter_directional in ['anisotropic']:
        return transformer.build_aniso(expr)
