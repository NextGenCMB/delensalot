<<<<<<< HEAD
import numpy as np
=======
from lenscarf.lerepi.core.metamodel.dlensalot import *

>>>>>>> 5c89a4d (refactor validator)

class analysis:
    def key(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def version(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def simidxs_mf(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def TEMP_suffix(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lens_res(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def zbounds(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def zbounds_len(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def pbounds(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class chaindescriptor:
    def p0(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p1(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p2(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p3(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p4(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p5(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p6(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def p7(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class computing:
    def OMP_NUM_THREADS(instance, attribute, value):
        if type(value) != int:
            raise ValueError('Must be int')


class data:
    def simidxs(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def class_parameters(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def package_(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def module_(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def class_(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def data_type(instance, attribute, value):
<<<<<<< HEAD
        desc = ['alm', 'map']
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def data_field(instance, attribute, value):
        desc = ['qu', 'eb']
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
=======
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def data_field(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
>>>>>>> 5c89a4d (refactor validator)


    def beam(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def nside(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def transferfunction(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class filter:
    def directional(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def data_type(instance, attribute, value):
<<<<<<< HEAD
        desc = ['alm', 'map']
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax(instance, attribute, value):
=======
>>>>>>> 5c89a4d (refactor validator)
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax_len(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_len(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax_unl(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_unl(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class itrec: 
    def tasks(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def simidxs(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 76828a1 (refactor)
    def simidxs_mf(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


<<<<<<< HEAD
=======
>>>>>>> 5c89a4d (refactor validator)
=======
>>>>>>> 76828a1 (refactor)
    def itmax(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def cg_tol(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lenjob_geometry(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

        
    def lenjob_pbgeometry(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def iterator_typ(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mfvar(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def soltn_cond(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def stepper(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax_plm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_plm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


<<<<<<< HEAD
<<<<<<< HEAD
=======
    def lmax_filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


>>>>>>> 5c89a4d (refactor validator)
=======
>>>>>>> 76828a1 (refactor)
class job:
    def QE_lensrec(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def MAP_lensrec(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def inspect_result(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def map_delensing(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def build_OBD(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class mapdelensing:
    def edges(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def dlm_mod(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def iterations(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def masks(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def Cl_fid(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

        
    def libdir_it(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def binning(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def spectrum_calculator(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


<<<<<<< HEAD
    def dir_btempl(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class meta:
    def version(instance, attribute, value):
        if type(value) != str:
            raise ValueError('Must be str')
=======
class meta:
    def version(instance, attribute, value):
        if type(value) != str:
<<<<<<< HEAD
            raise ValueError('Must be int')
>>>>>>> 5c89a4d (refactor validator)
=======
            raise ValueError('Must be str')
>>>>>>> 76828a1 (refactor)
        desc = ['0.9']
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class model:
    def meta(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Meta
        desc_type = [DLENSALOT_Meta]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


    def computing(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Computing
        desc_type = [DLENSALOT_Computing]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


    def job(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Job
        desc_type = [DLENSALOT_Job]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


    def analysis(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Analysis
        desc_type = [DLENSALOT_Analysis]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


    def data(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Data
        desc_type = [DLENSALOT_Data]
<<<<<<< HEAD
        assert attribute.type in desc_type, TypeError('Must be in {}, but is {}'.format(desc_type, attribute.type))
        
        if value.data_type == 'alm':
            assert value.data_field == 'eb', ValueError("I don't think you have qlms, ulms. More likely, you have E/B maps, Q/U maps, or e/b lms.")

        # lmax and transferfunction must have same length
        assert len(value.class_parameters['cl_transf'])==value.lmax+1, ValueError("Transferfunction length {} must be equal to 'lmax', but is {}".format(value.class_parameters['cl_transf'], value.lmax))
=======
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))
>>>>>>> 5c89a4d (refactor validator)


    def noisemodel(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Noisemodel
<<<<<<< HEAD
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_OBD
        desc_type = [DLENSALOT_Noisemodel]
        assert attribute.type in desc_type, TypeError('Must be in {}, but is {}'.format(desc_type, attribute.type))
    	# if truncated, OBD must be None
        if value.lowell_treat == 'trunc' or value.lowell_treat == None:
            assert value.OBD == None, TypeError("lowell_treat = {}: OBD is not used and should be set to None".format(value.lowell_treat))
        elif value.lowell_treat == 'OBD':
            assert value.OBD.type == DLENSALOT_OBD, TypeError("As lowell_treat = 'OBD': OBD must be {}".format(DLENSALOT_OBD))
=======
        desc_type = [DLENSALOT_Noisemodel]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))
>>>>>>> 5c89a4d (refactor validator)


    def qerec(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Qerec
        desc_type = [DLENSALOT_Qerec]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))

<<<<<<< HEAD
        for simidx in value.simidxs_mf:
            assert simidx in value.simidxs, ValueError('Meanfield simidx must be in {}, but is {}'.format(desc_type, attribute.type))

=======
>>>>>>> 5c89a4d (refactor validator)

    def itrec(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Itrec
        desc_type = [DLENSALOT_Itrec]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


    def madel(instance, attribute, value):
        from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Mapdelensing
        desc_type = [DLENSALOT_Mapdelensing]
        assert attribute.type in desc_type, ValueError('Must be in {}, but is {}'.format(desc_type, attribute.type))


class noisemodel:

    def lowell_treat(instance, attribute, value):
        desc = ['OBD', 'trunc', None]
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def OBD(instance, attribute, value):
<<<<<<< HEAD
        desc_dtype = [type(value), type(None)]
        assert type(value) in desc_dtype, ValueError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
=======
        desc = [type(value)]
        assert type(value) in desc, ValueError('Must be in {}, but is {}'.format(desc, type(value)))
>>>>>>> 5c89a4d (refactor validator)


    def lmin_tlm(instance, attribute, value):
        desc_min = 0
        desc_max = np.inf
        assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


    def lmin_elm(instance, attribute, value):
        desc_min = 0
        desc_max = np.inf
        assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


    def lmin_blm(instance, attribute, value):
        desc_min = 0
        desc_max = np.inf
        assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


    def nlev_t(instance, attribute, value):
        desc_dtype = [int, np.float, np.float64]
        desc = [0.,np.inf]
        assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
        assert desc[0]<value<desc[1], ValueError('Must be in {}, but is {}'.format(desc, value))


    def nlev_p(instance, attribute, value):
        desc_dtype = [int, np.float, np.float64]
        desc = [0.,np.inf]
        assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
        assert desc[0]<value<desc[1], ValueError('Must be in {}, but is {}'.format(desc, value))


    def rhits_normalised(instance, attribute, value):
        desc_dtype = [str, type(None)]
        assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
        if type(value) is not type(None):
            assert os.path.isfile(value), OSError("File doesn't exist: {}".format(value))


    def mask(instance, attribute, value):
        desc_dtype = [str, type(None)]
        assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
        if type(value) is not type(None):
            assert os.path.isfile(value), OSError("File doesn't exist: {}".format(value))


    def ninvjob_geometry(instance, attribute, value):
        desc_dtype = [str, type(None)]
        desc = ['healpix_geometry', 'healpix_geometry_qe', 'thin_gauss', 'pbdGeometry']
        assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class obd:
    def libdir(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def rescale(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def tpl(instance, attribute, value):
<<<<<<< HEAD
        desc = ['template_dense', None]
        assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
=======
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
>>>>>>> 5c89a4d (refactor validator)


    def nlev_dep(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class qerec:
    def simidxs(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def simidxs_mf(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def Lmin(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def qest(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def cg_tol(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

        
    def ninvjob_qe_geometry(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax_qlm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_qlm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


<<<<<<< HEAD
<<<<<<< HEAD
=======
    def lmax_filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_filter(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


>>>>>>> 5c89a4d (refactor validator)
=======
>>>>>>> 76828a1 (refactor)
    def chain(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def cl_analysis(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


class stepper:
    def typ(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def lmax_qlm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def mmax_qlm(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def xa(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


    def xb(instance, attribute, value):
        desc = [attribute]
        assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
