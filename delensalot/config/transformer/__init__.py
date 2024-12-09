from delensalot.config.visitor import transform, transform3d
from delensalot.core.iterator.iteration_handler import iterator_transformer
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm, DLENSALOT_Concept
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer 
from delensalot.core.handler import OBD_builder, Sim_generator, QE_lr, MAP_lr, Map_delenser
from delensalot.core.opfilt.opfilt_handler import QE_transformer, MAP_transformer, QE_iso_transformer, QE_aniso_transformer, MAP_iso_transformer, MAP_aniso_transformer

@transform3d.case(DLENSALOT_Model_mm, str, l2delensalotjob_Transformer)
def f1(expr, job_id, transformer): # pylint: disable=missing-function-docstring
    if "generate_sim" == job_id:
        return transformer.build_generate_sim(expr)
    if "build_OBD" == job_id:
        return transformer.build_OBD_builder(expr)
    if "QE_lensrec" == job_id:
        return transformer.build_QE_lensrec(expr)
    if "MAP_lensrec" == job_id:
        return transformer.build_MAP_lensrec(expr)
    if "delens" == job_id:
        return transformer.build_delenser(expr)
    if "analyse_phi" == job_id:
        return transformer.build_phianalyser(expr)
    else:
        assert 0, 'Dont understand your key: {}'.format(job_id)
    

@transform3d.case(DLENSALOT_Concept, str, l2delensalotjob_Transformer)
def f1(expr, job_id, transformer): # pylint: disable=missing-function-docstring
    if "generate_sim" == job_id:
        return transformer.build_generate_sim(expr)
    if "build_OBD" == job_id:
        return transformer.build_OBD_builder(expr)
    if "QE_lensrec" == job_id:
        return transformer.build_QE_lensrec(expr)
    if "MAP_lensrec" == job_id:
        return transformer.build_MAP_lensrec(expr)
    if "delens" == job_id:
        return transformer.build_delenser(expr)
    if "analyse_phi" == job_id:
        return transformer.build_phianalyser(expr)
    else:
        assert 0, 'Dont understand your key: {}'.format(job_id)

@transform.case(DLENSALOT_Concept, QE_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.qe_filter_directional in ['isotropic']:
        return transformer.build_iso(expr)
    elif expr.qe_filter_directional in ['anisotropic']:
        return transformer.build_aniso(expr) 

@transform.case(QE_lr, QE_iso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pee', 'pbb']:
        return transformer.build_opfilt_iso_p(expr)
    elif expr.k in ['ptt']:
        return transformer.build_opfilt_iso_t(expr)
    elif expr.k == 'p':
        assert 0, "implement if needed"
        # return transformer.build_opfilt_iso_tp(expr)
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

@transform.case(QE_lr, QE_aniso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pee', 'pbb']:
        return transformer.build_opfilt_aniso_p(expr)
    elif expr.k in ['ptt']:
        return transformer.build_opfilt_aniso_t(expr)
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

@transform.case(QE_lr, QE_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.qe_filter_directional in ['isotropic']:
        return transformer.build_iso(expr)
    elif expr.qe_filter_directional in ['anisotropic']:
        return transformer.build_aniso(expr)

@transform.case(MAP_lr, MAP_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.it_filter_directional in ['isotropic']:
        return transformer.build_iso(expr)
    elif expr.it_filter_directional in ['anisotropic']:
        return transformer.build_aniso(expr)

@transform.case(MAP_lr, MAP_iso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pbb']:
        return transformer.build_opfilt_iso_p(expr)
    elif expr.k == 'pee':
        return transformer.build_opfilt_iso_e(expr)
    elif expr.k == 'ptt':
        return transformer.build_opfilt_iso_t(expr)
    elif expr.k == 'p':
        return transformer.build_opfilt_iso_tp(expr)
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

@transform.case(MAP_lr, MAP_aniso_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.k in ['p_p', 'p_eb', 'p_be', 'peb', 'pbb']:
        return transformer.build_opfilt_aniso_p(expr)
    elif expr.k == 'pee':
        assert 0, "implement if needed"
    elif expr.k == 'ptt':
        return transformer.build_opfilt_aniso_t(expr)
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

@transform.case(MAP_lr, iterator_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.iterator_typ in ['constmf']:
        return transformer.build_constmf_iterator(expr)
    elif expr.iterator_typ in ['pertmf']:
        return transformer.build_pertmf_iterator(expr)
    elif expr.iterator_typ in ['fastWF']:
        return transformer.build_fastwf_iterator(expr)
    elif expr.iterator_typ in ['constmf_p']:
        return transformer.build_glm_constmf_iterator(expr)
    elif expr.iterator_typ in ['constmf_gc']:
        return transformer.build_gclm_constmf_iterator(expr)