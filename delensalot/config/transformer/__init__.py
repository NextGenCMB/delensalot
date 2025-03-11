from delensalot.config.visitor import transform, transform3d

from delensalot.config.metamodel.delensalot_mm import DELENSALOT_Model as DELENSALOT_Model_mm, DELENSALOT_Concept
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer as l2delensalotjob_Transformer


@transform3d.case(DELENSALOT_Concept, str, l2delensalotjob_Transformer)
def f1(expr, job_id, transformer): # pylint: disable=missing-function-docstring
    if "generate_sim" == job_id:
        return transformer.build_datacontainer(expr)
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


@transform3d.case(DELENSALOT_Model_mm, str, l2delensalotjob_Transformer)
def f1(expr, job_id, transformer): # pylint: disable=missing-function-docstring
    if "generate_sim" == job_id:
        return transformer.build_datacontainer(expr)
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