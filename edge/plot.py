import numpy as np
import matplotlib as mpl
#mpl.rcParams.update({'font.size': 17})
#mpl.rcParams.update({'font.family': 'serif'})
#mpl.rcParams.update({'text.usetex': True})
import matplotlib.pyplot as plt

from .utils import *
from ..constants import *


def plot_darklight_vs_edge_mstar(halo, t,z,vsmooth,sfh_insitu,mstar,mstar_insitu, zre=4., force_rmax_in_rvir=False,
                                 fn_vmax=None, figfn=None, plot_separately=False, legend=True, 
                                 sfh_lim=None, vmax_lim=None, mstar_lim=None):
    """
    Assumes that the given arrays t,vsmooth,sfh_insitu,mstar (and possibly
    mstar_insitu) are increasing in time.
    """

    tre = np.interp(zre,z[::-1],t[::-1])  # time of reionization, Gyr

    plot_scatter = False if mstar.ndim==1 else True

    if plot_scatter:
        sfh_stats          = np.array([ np.percentile(sfh_insitu  [:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(t)) ])
        mstar_insitu_stats = np.array([ np.percentile(mstar_insitu[:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(t)) ])
        mstar_stats        = np.array([ np.percentile(mstar       [:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(t)) ])
    
    # plotting preliminaries
    if not plot_separately:
        fig1 = plt.figure(figsize=(5,6.25))
        gs   = fig1.add_gridspec(ncols=1,nrows=7,hspace=0)
        ax1a = fig1.add_subplot(gs[:4,0])
        ax1b = ax1a.twinx()
        ax2  = fig1.add_subplot(gs[4:,0],sharex=ax1a)
        plt.setp(ax1a.get_xticklabels(),visible=False)
    else:
        fig1 = plt.figure(1)
        ax1a = fig1.add_subplot(111)
        ax1b = ax1a.twinx()
        fig2 = plt.figure(2)
        ax2  = fig2.add_subplot(111)

    # get halo data
    if fn_vmax==None:
        t_edge,z_edge,mstar_edge,rbins,menc_dm,r200c = halo.calculate_for_progenitors('t()','z()','M200c_stars','rbins_profile','dm_mass_profile','r200c')

        vmax_edge = np.zeros(len(t_edge))
        for i in range(len(t_edge)):
            vcirc = np.sqrt( G*menc_dm[i]/rbins[i] )
            vmax_edge[i] = max(vcirc) if not force_rmax_in_rvir else max(vcirc[ rbins[i]<r200c[i] ])
        tre = np.interp(zre,z_edge,t_edge)
    
        tsfh_edge_raw = np.arange(0,t[-1],0.02) # not midpoints, but left of bin
        try: sfh_edge_raw = halo.calculate('SFR_histogram')  # only take SFH at last time; dt = 0.02 Gyr
        except: sfh_edge_raw = np.zeros(len(tsfh_edge_raw)-1)
        mstar_edge_insitu = np.concatenate([[0],np.array([sum(sfh_edge_raw[:i]) for i in range(len(sfh_edge_raw)) ]) * (0.02*1e9)])  # multliply by dt
        sfh_edge = rebin_sfh(t, tsfh_edge_raw,sfh_edge_raw)
        #sfh_100myr = [ sum(sfh_edge[i*5:i*5+5])/5. for i in range(int(len(sfh_edge)/5)) ]  # rebin to 100 myr intervals
        #tsfh_100myr = arange(0.05,0.1*len(sfh_100myr),0.1)
    else:
        t_edge,z_edge,vmax_edge = np.loadtxt(fn_vmax,unpack=True)
        t_edge,z_edge,vmax_edge = t_edge[::-1],z_edge[::-1],vmax_edge[::-1] # expects them to be in backwards time order


    # plot the vmaxes
    ax1a.plot(t,vsmooth,'C0',alpha=0.8,label='DarkLight')
    ax1a.plot(t_edge,vmax_edge,color='0.7',label='EDGE')
    ylims = ax1a.get_ylim() if vmax_lim==None else vmax_lim # [4,36]
    ax1a.plot(tre*np.ones(2),ylims,'k--')
    ax1a.set_ylim(ylims)

    # plot the SFHs
    dt = t[1:] - t[:-1]
    if plot_scatter:
        ax1b.bar(t[:-1],sfh_stats[:-1,1],alpha=0.25,width=dt,color='C0',align='edge',label='DarkLight')
    else:
        ax1b.bar(t[:-1],sfh_insitu[:-1],alpha=0.25,width=dt,color='C0',align='edge',label='DarkLight')
    if fn_vmax==None:  ax1b.bar(t[:-1],sfh_edge,alpha=0.25,width=dt,color='k',align='edge',label='EDGE')
    ax1b.axvline(tre,color='k',linestyle='--')


    # plot the mstar trajectories
    if plot_scatter:
        ax2.fill_between(t,mstar_stats[:,0],mstar_stats[:,2],color='C0',alpha=0.2)
        ax2.fill_between(t,mstar_stats[:,3],mstar_stats[:,4],color='C0',alpha=0.1)
        ax2.plot(t,mstar_stats[:,1],'C0',label='DarkLight')
        ax2.plot(t,mstar_insitu_stats[:,1],'C0',alpha=0.3)
    else:
        ax2.plot(t,mstar,'C0',label='DarkLight')
        ax2.plot(t,mstar_insitu,'C0',alpha=0.3)

    if fn_vmax==None:
        ax2.plot(t_edge,mstar_edge,color='k',label='EDGE')
        ax2.plot(tsfh_edge_raw,mstar_edge_insitu,color='0.7')

    ax2.axvline(tre,color='k',linestyle='--')

        
    # finishing touches
    ax1a.set_ylim(ylims)
    ax1a.set_ylabel(r'v$_{\rm max}$ (km/s)')

    ax1b.set_yscale('log')
    if sfh_lim != None:  ax1b.set_ylim(sfh_lim)  # [1e-6,2e-2]
    ax1b.set_xlim([0,14])
    ax1b.set_ylabel('SFH (M$_\odot$/yr)')
    if legend: ax1b.legend(loc=4,frameon=False) #'best'
    
    ax2.set_yscale('log')
    if mstar_lim != None:  ax2.set_ylim(mstar_lim)  # [5e2,1e7]
    ax2.set_xlim([0,14])
    ax2.set_xlabel('t (Gyr)') 
    ax2.set_ylabel('M$_*$ (M$_\odot$)')
    if legend: ax2.legend(loc='best',frameon=False)

    
    if not plot_separately:

        if figfn==None:  return fig1, (ax1a,ax1b,ax2)
        
        try: fig1.tight_layout()
        except: print('error when tried tight_layout()!')
        try: 
            plt.savefig(figfn if figfn != None else 'darklight_vs_edge.pdf')
            print('wrote',figfn)
        except: print('error when tried to savefig, skipping!')

    else:

        if figfn==None:  return (fig1,fig2), (ax1a,ax1b,ax2)
        
        plt.figure(1)
        plt.savefig((figfn if figfn != None else 'darklight_vs_edge')+'-vmax_sfh.pdf')
        print((figfn if figfn != None else 'darklight_vs_edge')+'-vmax_sfh.pdf')

        plt.figure(2)
        plt.savefig((figfn if figfn != None else 'darklight_vs_edge')+'-mstar.pdf')
        print((figfn if figfn != None else 'darklight_vs_edge')+'-mstar.pdf')
