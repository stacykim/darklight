import os
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'text.usetex': True})
import matplotlib.pyplot as plt
from .constants import *



def get_shortname(simname):
    """
    Returns the halo number and a shortened version of the simulation name,
    e.g. giving 'Halo600_fiducial_later_mergers' returns '600' and '600lm'.
    """
    
    split = simname.split('_')
    shortname = split[0][4:]
    halonum = shortname[:]
    if len(split) > 2:
        if   halonum=='332': shortname += 'low'
        elif halonum=='383': shortname += 'late'
        elif halonum=='600': shortname += 'lm'
        elif halonum=='624': shortname += 'hm'
        elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
        elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
        elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
        else:
            print('unsupported simulation',simname,'! Not sure what shortname to give it. Aborting...')
            exit()
    elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'
    #DMOname = 'Halo'+halonum+'_DMO' if split[-1]=='fiducial' else None

    return halonum, shortname



def load_tangos_data(simname,machine='astro'):

    import tangos
    
    halonum, shortname = get_shortname(simname)

    if machine=='astro':
        tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
        tangos_path_chimera  = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
        tangos_path = tangos_path_chimera if halonum=='383' else tangos_path_edge
    else:
        print('support for machine',machine,'not implemented!')
        exit()

    tangos.core.init_db(tangos_path+'Halo'+halonum+'.db')
    sim = tangos.get_simulation(simname)

    return sim



def load_pynbody_data(simname,output=-1,machine='astro'):
    """
    Returns the particle data for the given simulation and output number.
    By default, returns the z=0 output, which is specified via '-1'.
    """
    
    import pynbody
    
    halonum, shortname = get_shortname(simname)

    if machine=='astro':

        pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
        pynbody_path_edgeGMs = '/vol/ph/astro_data2/shared/morkney/EDGE_GM/'
        pynbody_path_chimera = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'

        if halonum=='383':
            pynbody_path = pynbody_path_chimera
        elif halonum != shortname:
            pynbody_path = pynbody_path_edgeGMs
        else:
            pynbody_path = pynbody_path_edge
            
    else:

        print('support for machine',machine,'not implemented!')
        exit()

    if not os.path.isdir(pynbody_path+simname):
        print('full hydro particle data does not exist!')
        exit()

    if output == -1:
        # get all the outputs and grab the highest numbered one
        snapshots = glob.glob(os.path.join(pynbody_path,simname,'output_*'))
        snapshots.sort()
        simfn = os.path.join(pynbody_path,simname,snapshots[-1])
    else:
        simfn = os.path.join(pynbody_path,simname,'output_'+str(output).zfill(5))

    try:  particles = pynbody.load(simfn)
    except:
        print('particle data exists but failed to load given output!')
        print('attempted read of',simfn)
        exit()

    print('read',simfn)
    return particles



def rebin_sfh(t_new, t_old, sfh_old):
    """
    t_new and t_old must be in ascending order. Treats t_new and t_old as
    bin edges for calculating SFH values, so len(t_new) = len(sfh_new) + 1.
    If t_new[-1] > t_old[-1], then assumes SFR = 0 where no SFH exists.
    """
    assert np.all(t_new[:-1] <= t_new[1:]), 'rebin_sfh: requires elelments of t_new to be in ascending order!'
    assert np.all(t_old[:-1] <= t_old[1:]), 'rebin_sfh: requires elelments of t_old to be in ascending order!'

    if np.array_equiv(t_new,t_old):
        return sfh_old
    

    dt_old = t_old[1:] - t_old[:-1]
    mstar_old = sfh_old * dt_old # in whatever time units SFH is given (cancels out later)
    
    indicies = np.digitize(t_new, t_old)  # gives i s.t. bins[i-1] <= x < bins[i]
    sfh_new = np.zeros(len(t_new)-1)
    for i in range(len(t_new)-1):
        #print('\non t_new element',i)
        #print('t_new',t_new[i],'to',t_new[i+1])
        iold0 = indicies[i  ]
        iold1 = indicies[i+1]
        frac0 = (t_old[iold0] - t_new[i]) / dt_old[iold0-1]
        #print('falls in intervals iold0',iold0,'=',t_old[iold0-1],'to',t_old[iold0],'frac0',frac0)
        if iold1 >= len(t_old):
            iold1 = len(t_old) - 1
            frac1 = 1.
        else:
            frac1 = (t_new[i+1] - t_old[iold1-1]) / dt_old[iold1-1]
        #print('falls in intervals iold1',iold1,'=',t_old[iold1-1],'to',t_old[iold1],'frac1',frac1)

        #sfh_new[i] = (frac0*mstar_old[iold0-1] + sum(mstar_old[iold0:iold1]) + frac1*mstar_old[iold1-1]) / (t_new[i+1]-t_new[i])
        sfh_new[i] = (frac0*sfh_old[iold0-1] + sum(sfh_old[iold0:iold1]) + frac1*sfh_old[iold1-1]) / (frac0 + frac1 + iold1-iold0)
        #print('sfh_old')
        #print(sfh_old[iold0-1],frac0,'-->',frac0*mstar_old[iold0-1])
        #for j in range(iold0,iold1): print(sfh_old[j],1.,mstar_old[j])
        #print(sfh_old[iold1-1],frac1,frac1*mstar_old[iold1-1])
        #print('sfh_new')
        #print(sfh_new[i])
        
        #if i==5: exit()
        
    #print('last new timepoint',t_new[-1],'gyr')
    #print('old mstar',sum(mstar_old*GYR))
    dt_new = t_new[1:] - t_new[:-1]
    #print('new mstar',sum(sfh_new*dt_new*GYR))
    return sfh_new
    

def plot_darklight_vs_edge_mstar(halo, t,z,vsmooth,sfh_insitu,mstar,mstar_insitu, zre=4., figfn=None):
    """
    Assumes that the given arrays t,vsmooth,sfh_insitu,mstar (and possibly
    mstar_insitu) are increasing in time.
    """

    tre = np.interp(zre,z[::-1],t[::-1])  # time of reionization, Gyr

    plot_scatter = False if mstar.ndim==1 else True
    
    # plotting preliminaries
    fig1 = plt.figure(figsize=(5,6.25))
    gs   = fig1.add_gridspec(ncols=1,nrows=7,hspace=0)
    ax1a = fig1.add_subplot(gs[:4,0])
    ax1b = ax1a.twinx()
    ax2  = fig1.add_subplot(gs[4:,0],sharex=ax1a)
    plt.setp(ax1a.get_xticklabels(),visible=False)


    # get halo data
    t_edge,z_edge,mstar_edge,rbins,menc_dm = halo.calculate_for_progenitors('t()','z()','M200c_stars','rbins_profile','dm_mass_profile')
    vmax_edge = np.array([ np.sqrt(max( G*menc_dm[i]/rbins[i] )) for i in range(len(t_edge))])
    tre = np.interp(zre,z_edge,t_edge)
    
    tsfh_edge_raw = np.arange(0,t[-1],0.02) # not midpoints, but left of bin
    try: sfh_edge_raw = halo.calculate('SFR_histogram')  # only take SFH at last time; dt = 0.02 Gyr
    except: sfh_edge_raw = np.zeros(len(tsfh_edge_raw)-1)
    mstar_edge_insitu = np.concatenate([[0],np.array([sum(sfh_edge_raw[:i]) for i in range(len(sfh_edge_raw)) ]) * (0.02*1e9)])  # multliply by dt
    sfh_edge = rebin_sfh(t, tsfh_edge_raw,sfh_edge_raw)
    #sfh_100myr = [ sum(sfh_edge[i*5:i*5+5])/5. for i in range(int(len(sfh_edge)/5)) ]  # rebin to 100 myr intervals
    #tsfh_100myr = arange(0.05,0.1*len(sfh_100myr),0.1)


    # plot the vmaxes
    ylims = [4,36]
    ax1a.plot(t,vsmooth,'C0',alpha=0.8,label='DarkLight')
    ax1a.plot(t_edge,vmax_edge,color='0.7',label='EDGE')
    ax1a.plot(tre*np.ones(2),ylims,'k--')

    # plot the SFHs
    dt = t[1:] - t[:-1]
    if plot_scatter:
        ax1b.bar(t[:-1],sfh_insitu[:-1,1],alpha=0.25,width=dt,color='C0',align='edge',label='DarkLight')
    else:
        ax1b.bar(t[:-1],sfh_insitu[:-1],alpha=0.25,width=dt,color='C0',align='edge',label='DarkLight')
    ax1b.bar(t[:-1],sfh_edge,alpha=0.25,width=dt,color='k',align='edge',label='EDGE')
    ax1b.axvline(tre,color='k',linestyle='--')


    # plot the mstar trajectories
    if plot_scatter:
        ax2.fill_between(t,mstar[:,0],mstar[:,2],color='C0',alpha=0.2)
        ax2.fill_between(t,mstar[:,3],mstar[:,4],color='C0',alpha=0.1)
        ax2.plot(t,mstar[:,1],'C0',label='DarkLight')
        ax2.plot(t,mstar_insitu[:,1],'C0',alpha=0.3)
    else:
        ax2.plot(t,mstar,'C0',label='DarkLight')
        ax2.plot(t,mstar_insitu,'C0',alpha=0.3)

    ax2.plot(t_edge,mstar_edge,color='k',label='EDGE')
    ax2.plot(tsfh_edge_raw,mstar_edge_insitu,color='0.7')
    ax2.axvline(tre,color='k',linestyle='--')

        
    # finishing touches
    ax1a.set_ylim(ylims)
    ax1a.set_ylabel(r'v$_{\rm max}$ (km/s)')

    ax1b.set_yscale('log')
    ax1b.set_ylim([1e-6,2e-2])
    ax1b.set_xlim([0,14])
    ax1b.set_ylabel(r'SFH (M$_\odot$/yr)')
    ax1b.legend(loc='best')

    ylims = [5e2,1e7] # ax2.get_ylim() if not PLOT_MULTICOL else [5e2,1e7]
    ax2.set_yscale('log')
    ax2.set_ylim(ylims)
    ax2.set_xlim([0,14])
    ax2.set_xlabel('t (Gyr)') 
    ax2.set_ylabel(r'M$_*$ (M$_\odot$)')
    ax2.legend(loc='best')

    fig1.tight_layout()
    plt.savefig(figfn if figfn != None else 'darklight_vs_edge.pdf')
    print('wrote',figfn)


