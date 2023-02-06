import numpy as np
import abc
import os, sys
import hashlib
from os.path import join as opj
import importlib.util as iu
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import healpy as hp
from astropy.io import fits

import plancklens
from plancklens import utils, qresp, n0s
from plancklens.sims import planck2018_sims




from MSC import pospace as ps
from component_separation.cs_util import Config


class plot_helper():
    def __init__(self):
        self.csu = Config('Planck')
        self.colors = csu.CB_color_cycle
        self.colors_lt = csu.CB_color_cycle_lighter

        self.nside = 2048
        self.lmax_cl = 2048
        self.lmax_plot = 195
        self.lmax_qlm = 2500
        self.lmax_mask = 4096

        self.bk14_edges = np.array([2,55,90,125,160,195,230,265,300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]) # BK14 is [ 55  90 125 160 195 230 265 300], from bk14 = h5py.File('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/likedata_BK14.mat', 'r')['likedata']['lval'][0,:]
        self.ioreco_edges = np.array([2,30,200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
        self.cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])

        self.ll = np.arange(0,self.lmax_plot+1,1)
        self.binspace = 5
        self.scale_uk = (2 * self.ll + 1) * self.ll**2 * (self.ll + 1)**2
        self.scale_ps = self.ll*(self.ll+1)/(2*np.pi)
        self.label_scale_ps = r'$\frac{\ell(\ell+1)}{2 \pi}$'
        self.label_scale_lp = r'$\frac{\ell^2(\ell+1)^2}{2 \pi}$'
        self.scale_lp = self.ll**2 * (self.ll + 1)**2 * 1e7 / (2 * np.pi)

        self.psl = r'$\frac{l(l+1)}{2\pi}C_\ell \/ [\mu K^2]$'

        import matplotlib.colors as mcolors
        colors1 = plt.cm.Greys(np.linspace(0., .5, 128))
        colors2 = [plt.cm.Blues(np.linspace(0.6, 1., 128)), plt.cm.Reds(np.linspace(0.8, 1., 128)), plt.cm.Wistia(np.linspace(0.4, 1., 128)), plt.cm.Greens(np.linspace(0.6, 1., 128))]
        mymap = []
        nlevels_loc = [1.2, 2., 5.0, 50.0]
        for ni, n in enumerate(nlevels_loc):
            colors2[ni][:,-1] = 0.5
            colors_loc = np.vstack((colors1, colors2[ni]))
            mymap.append(mcolors.LinearSegmentedColormap.from_list('my_colormap', colors_loc))


        beam = 2.3
        lmax_transf = 4000
        transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)

        cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
        cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))
        clc_templ = cls_len['bb']


    def clamp(val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val


    def colorscale(hexstr, scalefactor):
        """
        Scales a hex string by ``scalefactor``. Returns scaled hex string.

        To darken the color, use a float value between 0 and 1.
        To brighten the color, use a float value greater than 1.

        >>> colorscale("#DF3C3C", .5)
        #6F1E1E
        >>> colorscale("#52D24F", 1.6)
        #83FF7E
        >>> colorscale("#4F75D2", 1)
        #4F75D2
        """

        hexstr = hexstr.strip('#')

        if scalefactor < 0 or len(hexstr) != 6:
            return hexstr

        r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

        r = clamp(r * scalefactor)
        g = clamp(g * scalefactor)
        b = clamp(b * scalefactor)

        return "#%02x%02x%02x" % (int(r), int(g), int(b))
        
        
    def get_ms(self, dat, binspace=5, bin_multipole=False):
        x = self.edges_center
        if bin_multipole:
            return get_weighted_avg(np.mean(dat, axis=0), np.std(dat, axis=0), binspace=binspace)
        else:
            return x, np.mean(dat, axis=0), np.std(dat, axis=0)


    def get_weighted_avg(self, mean, std, binspace):
        lscan = np.arange(0,len(mean),binspace)
        w_average = np.zeros(shape=int(round((len(mean)/binspace))))
        w_variance = np.zeros(shape=int(round((len(mean)/binspace))))
        for n in range(len(w_average)):
            w_average[n] = np.average(mean[n*binspace:(n+1)*binspace], weights=std[n*binspace:(n+1)*binspace])
            w_variance[n] = np.average(std[n*binspace:(n+1)*binspace])
            
            # w_variance[n] = np.average((mean-w_average[n])[n*binspace:(n+1)*binspace]**2, weights=std[n*binspace:(n+1)*binspace])
        return lscan, w_average, w_variance


    def bandpass_alms(self, alms, lmin, lmax=None):
        """
        lmin: minimum multipole to keep in alms
        lmax: maximimum multipole to keep in alms
        """
        
        if len(alms) == 3:
            out = np.zeros(alms.shape, dtype=complex)
            for idx, _alms in enumerate(alms):
                out[idx] = bandpass_alms(_alms, lmin, lmax=lmax)
            return out
        
        lmax_in_alms = hp.Alm.getlmax(len(alms))
        if lmax is None:
            lmax = lmax_in_alms
        else:
            assert isinstance(lmax, int), "lmax should be int: {}".format(lmax)
            assert lmax <= lmax_in_alms, "lmax exceeds lmax in alms: {} > {}".format(lmax, lmax_in_alms)
        
        fl = np.zeros(lmax_in_alms + 1, dtype=float)
        fl[lmin:lmax+1] = 1
        
        return hp.almxfl(alms, fl)


    def get_std(self, bcl_L):
        binspace=1
        for fgi, fg in enumerate(fgs):
            for nlevi, nlev in enumerate(nlevels):
                for iti, it in enumerate(range(len(bcl_L))):
                    x, mean, std = get_ms(bcl_L[iti,nlevi,fgi]*ct, binspace=binspace)
                    bcl_var[iti,fgi,nlevi] = std


    def load_plotdata(self):
        '''
        Most important settings
        '''
        fg = '00'
        survey='08b'
        run_lib = survey+'_%s_OBD_MF100_example'%fg
        dir_root = '/global/cscratch1/sd/sebibel/cmbs4/%s/'%run_lib

        simids = np.arange(0,99)
        # simids = np.delete(simids,[0, 40, 80])
        iterations = [12]
        blm_suffix = '' # 
        version = 'mf07'
        nlevels = [1.2,2,5,50]#,10.,100.]#[2.,5., 10., 100.]# [50.0]# [1.2, 1.5, 1.7, 2., 2.3, 5.0, 10.0, 50.0]
        cmbs4_edges2 = np.array([2, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 500, 1000, 2000, 3000])
        fgs = [fg] #, '07', '09']
        edges = cmbs4_edges
        '''
        /Most important settings
        '''

        if edges is bk14_edges:
            binid = 'bk14'
        if edges is ioreco_edges:
            binid = 'reionisation and recombination'
        if edges is cmbs4_edges:
            binid = 'cmbs4'
        if edges is cmbs4_edges2:
            binid = 'cmbs4_2'
        print('binid: {}'.format(binid))
        edges_center = (edges[1:]+edges[:-1])/2

        if version != '':
            vers_str = '/{}'.format(version)
        else:
            vers_str = ''

        ct = clc_templ[np.array(edges_center,dtype=int)]
        sha = hashlib.sha256()
        sha.update((str(edges)+str(blm_suffix)).encode())
        dirid = sha.hexdigest()[:4]

        bcl_cs_mfvar, bcl_L_mfvar  =  np.zeros(shape=(len(iterations)+2, len(fgs), len(nlevels), len(simids), len(edges)-1)), np.zeros(shape=(len(iterations)+2, len(fgs), len(nlevels), len(simids),len(edges)-1))
        for fgi, fg in enumerate(fgs):
            # run_lib = 'caterinaILC_May12_%s_OBD'%fg
            # dir_root = '/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_%s/%s/'%(survey, run_lib)
            dirroot_loc = dir_root#%(fg) # TODO make this fg dependent
            print('fg {} ({}/{}) started..'.format(fg, fgi+1, len(fgs)))
            for simidx, simid in enumerate(simids):
                data =  np.load(dirroot_loc + 'plotdata{}/{}'.format(vers_str,dirid) + '/ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg))
                # data =  np.load(dirroot_loc + '{}'.format(dirid) + '/Lenscarf_plotdata_ClBB_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg))
                bcl_L[0,fgi,:,simidx] = data[0][0]
                bcl_cs[0,fgi,:,simidx] = data[1][0]

                bcl_L[1,fgi,:,simidx] = data[0][1]
                bcl_cs[1,fgi,:,simidx] = data[1][1]
                
                for iti, it in enumerate(iterations):
                    bcl_L[2+iti,fgi,:,simidx] = data[0][2+iti]
                    bcl_cs[2+iti,fgi,:,simidx] = data[1][2+iti]
                
        print('dirid: {}'.format(dirid))
        print('loaded: {}'.format(dirroot_loc + 'plotdata{}/{}/'.format(vers_str,dirid)))


    def plot_delensed_spectra(self):
        lmax_plot = 200
        label_fQE = r'$_{L}B - \hat{B}^{QE}$'
        label_fMAP = r'$_{L}B - \hat{B}^{MAP}$'
        label_csQE = r'$_{C}B - \hat{B}^{QE}$'
        label_csMAP = r'$_{C}B - \hat{B}^{MAP}$'

        label_f = r'$_{{L}}C_l$'
        label_cs = r'$_{{C}}C_l$'

        label_QE = r'$_{{{}}}\hat{{C}}_{{l}}^{{BB,QE,del}}$'
        label_MAP = r'$_{{{}}}\hat{{C}}_{{l}}^{{BB,MAP,del}}$'
        cmap = matplotlib.cm.get_cmap('Paired')
        colors2 = np.array([cmap(nlevi/(6*len(simids))) for nlevi in range(6*len(simids))])
        binspace=1

        def remove_bin(bins, binidx):
            buff = bins[binidx+1]
            bins[binidx] = buff
            return bins
            
        def plot_content(a, bcl_cs, bcl_L, baseBmap, nlevi, fgi, iti=0, color=''):
            ms = 10
            alpha = 1.#(nlevi+1)/(len(nlevels_loc)+1)
            if color =='':
                label_suffix = ''
                if fgi == 0:
                    col = colorscale(colors[nlevi], 1.3)
                elif fgi == 1:
                    col = colorscale(colors[nlevi], 1.0)
                elif fgi == 2:
                    col = colorscale(colors[nlevi], 0.6)
            else:
                col = 'grey'
                label_suffix = ' mf-sub'
                
            xshiftQE = 1
            xshiftMAP = 0      
            if baseBmap == 'L':
                c1, c2 = 10, 7
                c3, c4 = 0, 2  
        #         x, mean, std = get_ms(bcl_L[0]*ct, binspace=binspace)
        #         a.fill_between(x, mean+std, mean-std, 
        #          label=label_f, color='black', alpha=0.2, edgecolor='black')
        #         a.plot(x, mean, color='black', alpha=0.6, lw=1)
                
        #         x, mean, std = get_ms(bcl_L[1]*ct, binspace=binspace)
        #         a.fill_between(x, mean+std, mean-std, label=label_QE.format(baseBmap)+label_suffix, color=col, alpha=0.3, edgecolor='black')
        #         a.plot(x, mean, color='grey', alpha=0.4, lw=1)
                
        #         x, mean, std = get_ms(bcl_L[2+iti]*ct, binspace=binspace)
        #         a.fill_between(x, mean+std, mean-std, label=label_MAP.format(baseBmap)+label_suffix, color=col, alpha=0.6, edgecolor='black')
        #         a.plot(x, mean, color='grey', alpha=0.9, lw=1)
                
                x, mean, std = get_ms(bcl_L[0]*ct, binspace=binspace)
                a.errorbar(x, mean, xerr=14, yerr=std, label=label_f, color='black', alpha=0.8, ls='', capthick=2, capsize=4, mfc='black', ms=6, fmt='o')

                x, mean, std = get_ms(bcl_L[1]*ct, binspace=binspace)
                a.errorbar(x+xshiftQE, mean, xerr=14, yerr=std, label=label_QE.format(baseBmap)+label_suffix, ls='', color='black', alpha=0.4, capthick=2, capsize=4, mec=col, mfc=col, ms=6, fmt='o')
                
                x, mean, std = get_ms(bcl_L[-1]*ct, binspace=binspace)
                a.errorbar(x+xshiftMAP, mean, xerr=14, yerr=std, label=label_MAP.format(baseBmap)+label_suffix, ls='', color='black', alpha=0.9, capthick=2, capsize=4, mec=col, mfc=col, ms=6, fmt='o')
                
            elif baseBmap == 'C':
                c1, c2 = 10, 7
                c3, c4 = 0, 2
                
                x, mean, std = get_ms(bcl_cs[0]*ct, binspace=binspace)
                a.errorbar(x, mean, xerr=14, yerr=std, label=label_f, color='black', alpha=0.8, ls='', capthick=2, capsize=3)
                    
                x, mean, std = get_ms(bcl_cs[1]*ct, binspace=binspace)
                a.errorbar(x+xshiftQE, mean, xerr=14, yerr=std, label=label_QE.format(baseBmap)+label_suffix, ls='', color=col, alpha=0.4, capthick=2, capsize=3)
                # a.fill_between(x, mean+std, mean-std, label=label_QE.format(baseBmap)+label_suffix, color=col, alpha=0.3, edgecolor='black')

                x, mean, std = get_ms(bcl_cs[-1]*ct, binspace=binspace)
                a.errorbar(x+xshiftMAP, mean, xerr=14, yerr=std, label=label_MAP.format(baseBmap)+label_suffix, ls='', color=col, alpha=0.9, capthick=2, capsize=3)
                # a.fill_between(x, mean+std, mean-std, label=label_MAP.format(baseBmap)+label_suffix, color=col, alpha=0.6, edgecolor='black')

        for fgi, fg in enumerate(fgs):
            for bBi, bB in enumerate(['L','C']):
                fig, ax = plt.subplots(4, 1, figsize=[12,24])
                label_f = r'$_{{L}}C_\ell$'
                label_cs = r'$_{{C}}C_\ell$'
                label_QE = r'$_{{{}}}\hat{{C}}_{{\ell}}^{{BB,QE,del}}$'
                label_MAP = r'$_{{{}}}\hat{{C}}_{{\ell}}^{{BB,MAP,del}}$'
                for nlevi, nlev in enumerate(nlevels_loc):
                    bcl_cs[:,fgi,nlevi_loc[nlevi]]
                    bcl_L[:,fgi,nlevi_loc[nlevi]]
                    baseBmap = bB
                    plot_content(
                        ax[nlevi],
                        bcl_cs[:,fgi,nlevi_loc[nlevi]], bcl_L[:,fgi,nlevi_loc[nlevi]],
                        baseBmap, nlevi, fgi)
        
                    ax[nlevi].set_xlim((30.,lmax_plot))
                    ax[nlevi].set_ylim((0.0e-6,4e-6))       
                    if baseBmap == 'L':
                        ax[nlevi].legend(title=r'$_{{{}}}C=$ B-lensing'.format(baseBmap), loc='upper right', framealpha=1.0)
                    else:
                        ax[nlevi].legend(title=r'$_{{{}}}C=$ ILC-map'.format(baseBmap), loc='upper right', framealpha=1.0)
                    # ax[nlevi].set_title('Nlev ratio: {}'.format(nlev))
                    if nlevi == len(nlevels_loc)-1:
                        ax[nlevi].set_xlabel(r'Multipole $\ell$', fontsize=22)
                    if bBi == 0:
                        ax[nlevi].set_ylabel(r'$C_\ell$ [$\mu K^2$]', fontsize=22)
                    if baseBmap == 'L':
                        plt.savefig('/global/homes/s/sebibel/notebooks/cmbs4/plots/'+survey+'/ClBlens{}_ens_avg_fg{}_allnlev_notitle.pdf'.format(version,fg), bbox_inches='tight')
                    else:
                        plt.savefig('/global/homes/s/sebibel/notebooks/cmbs4/plots/'+survey+'/ClBWf{}_ens_avg_fg{}_allnlev_notitle.pdf'.format(version,fg), bbox_inches='tight')
                plt.show()
        lmax_plot=200


    def plot_variance(self):
        colors2 = np.array([cmap(nlevi/(len(nlevels))) for nlevi in range(len(nlevels))])
        fig, ax = plt.subplots(1,2, figsize=[12, 6],sharex=True, sharey=True)
        plotlines = []
        plotlines_fg = []
        colbi = 11
        for fgi in range(len(fgs)):
            for nlevi in range(len(nlevels_loc)):
                ls = '-'
                if fgi == 0:
                    label = '{}'.format(nlevels_loc[nlevi])
                    #ls = '-'
                    col = colorscale(colors[nlevi], 1.3)
                    col_black = colorscale(colors[colbi], 1.3)
                elif fgi==1:
                    label = ''
                    #ls = '--'
                    col = colorscale(colors[nlevi], 1.0)
                    col_black = colorscale(colors[colbi], 1.0)
                else:
                    label = ''
                    #ls = ':'
                    col = colorscale(colors[nlevi], 0.6)
                    col_black = colorscale(colors[colbi], 0.6)

                # l1, = ax[0].plot(edges_center, bcl_var[fgi,1,nlevi_loc[nlevi]]/bcl_var[fgi,0,nlevi_loc[nlevi]],
                #                  color=col, ls = ls, label=label, lw=1)
                # l2, = ax[1].plot(edges_center, bcl_var[fgi,2,nlevi_loc[nlevi]]/bcl_var[fgi,0,nlevi_loc[nlevi]],
                #                  color=col, ls = ls, label=label, lw=3)
                
                # l0, = ax[0].plot(edges_center, bcl_var[fgi,0,nlevi], color=col, ls = ls, label=label, lw=3)
                
                l1, = ax[0].plot(edges_center, bcl_var[1,fgi,nlevi_loc[nlevi]], color=col, ls=ls, label=label, lw=3)
                l2, = ax[1].plot(edges_center, bcl_var[-1,fgi,nlevi_loc[nlevi]], color=col, ls=ls, label=label, lw=3)
                l3, = ax[1].plot([], [], color=col_black, ls=ls, label=None, lw=3)

                if fgi==1:
                    plotlines.append(l2)
                if nlevi == 0:
                    plotlines_fg.append(l3)
        ax[0].set_xlim(37.5,lmax_plot)
        ax[0].set_ylim(2e-8,1e-6)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Multipole, $\ell$', fontsize=18)
        ax[1].set_xlabel('Multipole, $\ell$', fontsize=18)
        ax[0].set_ylabel(r'$Var\left(_{{{}}}\hat{{C}}_{{l}}^{{BB,del}}\right)$', fontsize=18)
        handles, labels = ax[1].get_legend_handles_labels()
        fig.legend(plotlines, labels, loc='upper center', title='nlev mask ratio', bbox_to_anchor=[0.51, 0.9], framealpha=0.8)

        leg3 = plt.legend(plotlines_fg, [fg for fg in fgs], title='Fg-model', loc=1, bbox_to_anchor=[.12,0.4], framealpha=0.8)

        # plt.title('QE and MAP delensed ensemble variance. {} simulations.'.format(len(simids)) + '\n         QE'+' '*50+'MAP', x=-0.1)
        plt.savefig('/global/homes/s/sebibel/notebooks/cmbs4/plots/'+survey+'/ClBBQEMAP_var_allfg.pdf'.format(fg), bbox_inches='tight')


    def plot_rla(self):
        label_f_QE = r'$1-A_\ell^{QE}$'
        label_QE_L = r'$1-_{L}A_\ell^{QE}$'
        label_QE_C = r'$1-_{C}A_\ell^{QE}$'

        label_MAP_L = r'$1-_{L}A_\ell^{MAP}$'
        label_MAP_L_mfvar1 = r'$1-_{L}^{m07}A_\ell^{MAP}$'
        label_MAP_L_mfvar2 = r'$1-_{L}^{m00}A_\ell^{MAP}$'
        label_MAP_C = r'$1-_{C}A_\ell^{MAP}$'

        nlevi_loc = [0,1,2,3]
        nlevels_loc = np.take(nlevels,np.array(nlevi_loc))

        lmax_loc=256+1
        bl = edges[:-1]
        bu = edges[1:]

        blu1d = np.array([[l,u] for l,u in zip(bl,bu)]).flatten()
        delx = 40
        diffx = (bu - bl)/2
        xerr = diffx
        nrow, ncol = 2,2
        import string
        for fgi, fg in enumerate(fgs):
            fig, ax = plt.subplots(nrow,ncol,figsize=[8, 8], sharex=True, sharey=True)
            axs = ax.flat
            for n, axx in enumerate(axs): 
                axx.text(0.9, 1.02, string.ascii_uppercase[n], transform=axx.transAxes, 
                    size=20, weight='bold')
            plotlines = []
            plotlines_fg = []
            i=0.1
            for nlevi, nlev in enumerate(nlevels_loc):
                xshift = -delx/(len(nlevels)/2) + i*delx/(len(nlevels))
                if fgi == 0:
                    col_black = colorscale(colors[11], 1.3)
                    col = colorscale(colors[nlevi], 1.3)
                elif fgi == 1:
                    label_QE_L = r'$1-_{L}A_\ell^{QE}$'
                    label_MAP_L = r'$1-_{L}A_\ell^{MAP}$'
                    col_black = colorscale(colors[11], 1.0)
                    col = colorscale(colors[nlevi], 1.0)
                elif fgi == 2:
                    col_black = colorscale(colors[11], 0.6)
                    col = colorscale(colors[nlevi], 0.6)

                x, mean, std = get_ms(1-np.array([bcl_L[1,fgi,nlevi_loc[nlevi],simidx] for simidx in range(len(simids))]), binspace=1)
                ax[nlevi%nrow][int(nlevi/ncol)].errorbar(xshift+x, mean, yerr=std, xerr=xerr, mec=col, mfc=col, color='black', label=label_QE_L, alpha=0.2, lw=2, fmt='v', capsize=3, capthick=2, fillstyle='full')

                x, mean, std = get_ms(1-np.array([bcl_L[-1,fgi,nlevi_loc[nlevi],simidx] for simidx in range(len(simids))]), binspace=1)
                l2 = ax[nlevi%nrow][int(nlevi/ncol)].errorbar(xshift+x, mean, yerr=std, xerr=xerr, mec=col, mfc=col, color='black', label=label_MAP_L1, alpha=0.8, lw=2, fmt='o', capsize=3, capthick=2, fillstyle='full')
                l3 = ax[nlevi%nrow][int(nlevi/ncol)].errorbar(1, 1, yerr=1, xerr=1, label=None, mec=col_black, mfc=col_black, color='black',  alpha=0.8, lw=2, fmt='o', capsize=3, capthick=2, fillstyle='full')
                
                x, mean, std = get_ms(1-np.array([bcl_L[-1,fgi,nlevi_loc[nlevi],simidx] for simidx in range(len(simids))]), binspace=1)
                l2 = ax[nlevi%nrow][int(nlevi/ncol)].errorbar(xshift+x, mean, yerr=std, xerr=xerr, mec=col, mfc=col, color='grey', label=label_MAP_L, alpha=0.8, lw=2, fmt='o', capsize=3, capthick=2, fillstyle='full')
                
                # x, mean, std = get_ms(1-np.array([bcl_L[3,fgi,nlevi_loc[nlevi],simidx] for simidx in range(len(simids))]), binspace=1)
                # ax[nlevi%nrow][int(nlevi/ncol)].errorbar(1.1*xshift+x, mean, yerr=std, xerr=xerr, mec=col, mfc=col, color='purple', label=label_MAP_L, alpha=0.8, lw=2, fmt='o', capsize=3, capthick=2, fillstyle='full')
                if nlevi == 0:
                    plotlines_fg.append(l3)
                plotlines.append(l2)
                leg1 = ax[nlevi%nrow][int(nlevi/ncol)].legend(title='Fg 00'.format(fg), loc='lower right', framealpha=1.0, ncol=1, fontsize=14)
            label_QE_L, label_MAP_L = '',''
            ax[nlevi%nrow][int(nlevi/ncol)].legend(loc='lower right', framealpha=1.0, ncol=2, fontsize=18)
            ax[nlevi%nrow][int(nlevi/ncol)].set_xlim((2,lmax_plot))
            ax[nlevi%nrow][int(nlevi/ncol)].set_ylim((0.6,1))
            
            ax[0][1].set_xlim((30,200))
            ax[1][1].set_xlabel((30,200))
            ax[0][0].set_ylabel(r'$1-A_\ell$', fontsize=18)
            ax[1][0].set_ylabel(r'$1-A_\ell$', fontsize=18)
            ax[1][0].set_xlabel(r'Multipole, $\ell$', fontsize=18)
            ax[1][1].set_xlabel(r'Multipole, $\ell$', fontsize=18)
            # ax[0][1].set_title('Residual lensing amplitude, {} simulations\n'.format(len(simids)), x=-0.1)

            leg2 = plt.legend(plotlines, [nlev for nlev in nlevels_loc], title='nlev mask ratio', loc=1, ncol=2, bbox_to_anchor=[0.5,1.32])
            # leg3 = plt.legend(plotlines_fg, [f for f in [fg]], title='Fg-model', loc=1, bbox_to_anchor=[0.2,0.5])
            # plt.gca().add_artist(leg3)
            plt.gca().add_artist(leg2)
            plt.gca().add_artist(leg1)
            
            plt.savefig('/global/homes/s/sebibel/notebooks/cmbs4/plots/'+survey+'/rla_mfvar_fg00_07_allnlev.pdf'.format(fg))