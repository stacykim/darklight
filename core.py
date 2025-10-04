# darklight.py
# created 2020.03.20 by stacy kim

from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

from tangos.examples.mergers import *
from .constants import *

from . import DATA_DIR


##################################################
# DARKLIGHT

def smooth(t,y,tnew,sigma=0.5):
    """
    Smooths the given trajectory y(t) with a gaussian over timescales given by
    sigma, and returns the smoothed values at tnew.  Assumes t is in ascending order.
    """
    # smooth vmax rotation curve
    tgrid = arange(t[0],t[-1]+sigma,sigma)  # NOTE: interpolated in 500 Myr bins
    ygrid = interp(tgrid,t,y)#,left=0)
    ysmooth = filters.gaussian_filter1d(ygrid,sigma=1)
    return interp(tnew,tgrid,ysmooth)#,left=0)



def DarkLight(halo, DMO=True, vmax_file=None, n=1, 
              vthres='falling', zq=4., occupation=2.5e7,
              prequench='fiducial', postquench='schechter', postquench_scatter='increasing',
              nsc_ratio=0.7, t_delay=1,
              mapping='3bins', timesteps='sim', mergers=True, force_rmax_in_rvir=False):

    """
    Generates a star formation history, which is integrated to obtain the M* for
    a given halo. The vmax trajectory is smoothed before applying a SFH-vmax 
    relation to reduce temporary jumps in vmax due to mergers. Returns the
    timesteps t, z, the smoothed vmax trajectory, the in-situ star formation 
    history, the stellar mass formed in-situ, and the total stellar mass
    (i.e. includes accreted stars).  Each returned array except for the t and z
    arrays have n rows, one for each of n realizations.

    Notes on Inputs: 

    halo = tangos Halo object for which to compute properties.  If not available,
        can instead run on a vmax trajectory given in a file (see next).

    DMO = True if running on a DMO simulation.  Will then multiply particle
        masses by sqrt(1-fbary) and an additional suppression to match values in 
        full hydrodynamic sims, as measured in EDGE (see Kim et al. 2024)

    vmax_file = string or None.  Alternative method to supply a halo, if tangos Halo
        object is not available.  Can supply a file with name vmax_file that has 
        three columns: time (Gyr), redshift, vmax (km/s), with each row corresponding
        to a time step.  Expects time to be in increasing order.

    n = integer. The number of galaxy realizations to create.

    vthres = float or 'falling'.  The minimum vmax (in km/s) required for halos to
        start forming stars after reionization quenching.  By default uses the 
        EDGE1 value of 26.3 km/s.  The 'falling' model adopts a redshift-dependent
        vthres that falls after quenching to model the time it takes for cold gas
        to build up before rejuvenation in halos with M200 ~ few x 10^9 Msun (see
        Kim et al. 2024 for details).

    zq = the redshift when quenching occurs.  Assumes all galaxies immediately
        quench.  Accepts a float and by default, uses the EDGE1 value of zq=4.

    occupation = string or float.  Sets which halos have galaxies, given as a halo
        mass in solar masses, or one of the occupation functions below.  If a float, 
        halos with masses below are not allowed to host galaxies.  By default, 
        assumes 2.5e7 msun, based on the resolution limit of the EDGE sims.  

        The following occupation functions are supported.  These are derived from
        simulations or inferred from the Milky Way dwarfs.

        'all' = all halos DarkLight estimates have non-zero stellar mass is occupied
        'edge1' = occupation fraction derived from fiducial EDGE1 simulations
        'edge1rt' = occupation fraction from EDGE1 simulations with radiative 
            transfer; this is significantly higher than 'edge1'
        'nadler20' = from Nadler+ 2020's fit to the MW dwarfs. Note that this
            was parameterized in M200c, but we have made some simplifying 
            assumptions to convert it to vmax.

    prequench = the pre-quenching SFR-vmax relation to use.  By default, uses
        'fiducal' (see Kim et al. 2024).  See definition of sfr_pre() to see
        alternative methods.

    postquench = the post-quenching SFR-vmax relation to use.  By default,
        uses 'schechter'.  See definition of sfr_post() to see alternative methods.

    postquench_scatter = what scatter to assume in SFR-vmax relation after 
        reionization quenching

        'increasing' = adopts a scatter that's small for halos with large vmax,
            and increases towards smaller vmax
        'flat' = adopts a 1-sigma symmertic scatter of 0.3 dex

    mapping = how to apply the SFR-vmax relation.  The options are:

        'all' = maps vmax at each timestep into the SFR for that timestep
        '3bins' = the default, applies 
            (1) prequench relation used to map average vmax before quenching
                to derive single <SFR> for prequench period
            (2) quenched period (SFR=0) after zq while vmax < vthres
            (3) postquench relation used to derive a single SFR based on 
                vmax(z=0) for all timesteps after zq when vmax > vthres
        '2bins' = (1) like above before zq and (2) postquench relation
            after zq (even if vmax < vthres), again using single SFR
            based on vmax(z=0) for all timesteps

    timesteps = resolution of SFH, in Gyr, or 'sim' to use simulation timesteps.
        Used for both main and accreted halos.
    
    mergers = whether or not to include the contribution to M* from in-situ
        star formation, mergers, or both.

        True = include  M* of halos that merged with main halo
        False = only compute M* of stars that formed in-situ in main-halo
        'only' = only compute M* of mergers

    force_rmax_in_rvir = whether to require rmax to be within r200.  In the
        EDGE simulations, more accurately reprdouces hydrodynamic simulations
        if this is turned off (set to False).

    """

    assert (mergers=='only' or mergers==True or mergers==False), "DarkLight: keyword 'mergers' must be True, False, or 'only'! Got "+str(mergers)+'.'

    if halo==None:
        print('got halo==None !')
        return np.array([]), np.array([]), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n)


    # compute or read in vmax trajectory
    if vmax_file==None:

        t,z,rbins,menc_dm, m200c, r200c = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile', 'M200c', 'r200c')

        if len(t)==0: 
            return np.array([]), np.array([]), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n), np.array([])

        vmax = np.zeros(len(t))
        for i in range(len(t)):
            vcirc = np.sqrt( G*menc_dm[i]/rbins[i] )
            try: vmax[i] = max(vcirc) if not force_rmax_in_rvir else max(vcirc[ rbins[i]<r200c[i] ])  # make sure rmax < r200
            except ValueError as e:
                print(e)
                print('halo',halo.halo_number,'at t =',round(t[i],2),'Gyr. skipping!')
                return np.array([]), np.array([]), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n), np.array([[]]*n), np.array([])
        vmax *=  sqrt(1-FBARYON) if DMO else 1

    else:
        t,z,vmax = loadtxt(vmax_file,unpack=True)
        t,z,vmax = t[::-1],z[::-1],vmax[::-1] # change to backwards time order to match tangos

    
    ############################################################
    # Get values at points where DarkLight SFH will be calculated

    if timesteps == 'sim':
        tt = t[::-1]  # reverse arrays so time is increasing
        zz = z[::-1]
    else:
        tt = arange(t[-1],t[0],timesteps)
        zz = interp(tt, t[::-1], z[::-1])
    
    # make sure zq included in time steps, if halo exists then
    if zz[-1] < zq and zz[0] > zq:
        if zq not in zz:
            iq = where(zz<=zq)[0][0]
            tt = concatenate([ tt[:iq], [interp(zq,z,t)], tt[iq:] ])
            zz = concatenate([ zz[:iq], [zq],             zz[iq:] ])

    dt = tt[1:]-tt[:-1] # since len(dt) = len(t)-1, need to be careful w/indexing below

    if len(t) > 1:
        vsmooth = smooth(t[::-1],vmax[::-1],tt,sigma=0.5) # smoothed over 500 Myr
    else:
        vsmooth = vmax

    
    ############################################################
    # Generate the star formation histories

    nNSC=0
    
    # check if halo is occupied
    if vmax_file == None:
        m = halo['M200c'] if 'M200c' in halo.keys() else 1. # if no mass in tangos, then probably very low mass, give arbitrarily low value
        pocc = occupation_fraction(vsmooth[-1],m,method=occupation)
        occupied = np.random.rand(n) < pocc
    else:  # if just given file of vmaxes, then assume it is occupied
        occupied = np.ones(n)

    # compute in-situ component
    sfhs_insitu   = np.zeros((n,len(tt)))
    mstars_insitu = np.zeros((n,len(tt)))
    vsmooth_cored = np.zeros((n,len(tt)))

    for iis in range(n):
        if occupied[iis] and mergers != 'only':
            sfhs_insitu[iis],vsmooth_cored[iis] = sfh(tt,dt,zz,vsmooth,np.interp(tt, t[::-1],m200c[::-1]),
                                                      DMO=DMO,vthres=vthres,zq=zq,mapping=mapping,scatter=True,
                                                      prequench=prequench,postquench=postquench,postquench_scatter=postquench_scatter)
            mstars_insitu[iis] = np.array([0] + [ sum(sfhs_insitu[iis][:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ])

    # compute accreted component
    mstars_accreted = np.zeros((n,len(tt)))
    if mergers and sum(occupied)>0:

        zmerge, qmerge, hmerge, msmerge,nNSCmerge = accreted_stars(halo,vthres=vthres,zq=zq,timesteps=timesteps,occupation=occupation,DMO=DMO,
                                                         mapping=mapping,n=int(sum(occupied)),prequench=prequench,postquench=postquench,
                                                         postquench_scatter=postquench_scatter, nsc_ratio=nsc_ratio, t_delay=t_delay)

        # change from mstar for each merger -> cumsum(mstar) for each time
        mstars_accreted[occupied.astype(bool)] = np.array([np.sum(msmerge[zmerge>z],axis=0) for z in zz]).T 

        nNSC += check_nsc(qmerge, zmerge, hmerge, vsmooth, zz, tt, halo, vthres=vthres, zq=zq, nsc_ratio=nsc_ratio, t_delay=t_delay)
        nNSC += sum(nNSCmerge)
            
       
    # compute total stellar mass and we're done!
    mstars_tot = mstars_insitu + mstars_accreted
    return tt,zz,vsmooth_cored,sfhs_insitu,mstars_insitu,mstars_tot,nNSC




##################################################
# OCCUPATION FRACTION

# data for EDGE occupation fraction
vocc_bin_edges = array([2., 9., 16., 23., 30.])
vocc_edge    = sqrt(vocc_bin_edges[1:]*vocc_bin_edges[:-1])
focc_edge1   = array([0.033, 0.141, 0.250, 1.0])
focc_edge1rt = array([0.484, 0.591, 0.581, 1.0])

# data for Nadler+ 2018's occupation fraction
from colossus.halo.concentration import concentration
from colossus.halo.mass_defs import changeMassDefinition
from colossus.cosmology import cosmology
from colossus.utils.constants import G
cosmo = cosmology.setCosmology('planck18')  # can set different cosmology here
h0    = cosmo.Hz(0)/100 # normalized hubble constant, z=0
z=0

log10mvir, focc = np.loadtxt(DATA_DIR+'nadler2020-pocc.dat',unpack=True)
mvir = 10**log10mvir
cvir = concentration(mvir, 'vir', z, model='diemer19')
m200_div_h, r200_div_h, c200 = changeMassDefinition(mvir/h0, cvir, z, 'vir', '200c', profile='nfw')
m200_nadler,r200 = m200_div_h * h0, r200_div_h * h0
rs = r200/c200
xmax = 2.16258
rmax = xmax*rs
def fNFW(x):   return np.log(1+x) - x/(1+x)  # x = r/rs
vmax = np.sqrt(G*m200_nadler*fNFW(xmax)/fNFW(c200)/rmax)
vocc_nadler = np.concatenate([[2], vmax, [30]])
focc_nadler = np.concatenate([[0], focc, [ 1]])


def occupation_fraction(vmax,m200,method='edge1'):

    if isinstance(method, float):
        return 1 if m200 > method else 0
    elif method=='all':
        return 1.
    elif method=='edge1':
        return np.interp(vmax, vocc_edge, focc_edge1)
    elif method=='edge1rt':
        return np.interp(vmax, vocc_edge, focc_edge1rt)
    elif method=='nadler20':
        #return np.interp(vmax, vocc_nadler, focc_nadler)
        return np.interp(m200, m200_nadler, focc, left=0, right=1)
    else:
        raise ValueError('occupation fraction method '+method+' not recognized')



##################################################
# SFH ROUTINES

def vmax_rejuvenation(t,z,zq=4.):
    if hasattr(t,'__iter__'):
        vSF = np.zeros(len(t))
        vSF[z<=zq] = 13.5*np.exp(-t[z<=zq]/5)+23
        return vSF
    else:
        return 0 if z>zq else 13.5*np.exp(-t/5)+23


def sfr_pre(vmax,method='fiducial'):

    if not hasattr(vmax,'__iter__'):
        if vmax==0:  return 0
        v = vmax if vmax<=20 else 20
    else:
        v = vmax.copy()
        v[ v>20 ] = 20.

    if   method == 'fiducial': sfr = 10**(6.78*log10(v)-11.6)  # no turn over, simple log-linear fit to dataset below
    elif method == 'fiducial_with_turnover' :  sfr = 2e-7*(v/5)**3.75 * exp(v/5)  # with turn over at small vmax, SFR vmax calculated from halo birth, fit by eye
    else:  raise ValueError('Do not recognize sfr_pre method '+method)

    if hasattr(v,'__iter__'):  sfr[v==0] = 0

    return sfr


def sfr_post(vmax,method='schechter'):
    if   method == 'schechter'   :  return 7.06 * (vmax/182.4)**3.07 * exp(-182.4/vmax)  # schechter fxn fit
    elif method == 'schechterMW' :  return 6.28e-3 * (vmax/43.54)**4.35 * exp(-43.54/vmax)  # schechter fxn fit w/MW dwarfs
    elif method == 'linear'      :  return 10**( 5.48*log10(vmax) - 11.9 )  # linear fit w/MW dwarfs


def sfr_scatter(z, vmax, zq=4., prequench='fiducial', postquench='increasing'):
    """
    Returns scatter in the SFR-vmax relation.  Assumes 0.4 dex lognormal scatter
    before reionization quenching.  Postquench scatter is determined by given method.

    z and vmax must be arrays of the same length.

    Notes on Inputs:

    method = type of relation to adopt after reionization quenching

        'increasing' = scatter increases for smaller halos (lower vmax)
        'flat'       = 0.3 dex lognormal scatter (independent of mass)
    """

    if postquench=='increasing':  # increasing scatter for small vmax after quenching
        log10scatter = array([ 0.4 if zz > zq else (-0.651*log10(vv)+1.74) for zz,vv in zip(z,vmax) ])
        log10scatter[ log10scatter < 0.2 ] = 0.2 # max out at 0.2 dex at high-mass end, when extrapolating above fit
        return np.array([ 10**np.random.normal(0,log10s) for log10s in log10scatter ])
    else:
        return array([ 10**np.random.normal(0,0.4 if zz > zq else 0.3) for zz in z ])
    
    
def sfh(t, dt, z, vmax, m200, vthres=26.3, zq=4.,mapping='3bins',prequench='fiducial',postquench='schechter',
        postquench_scatter='increasing',scatter=False, DMO=False):

    """
    Assumes we are given a halo's entire vmax and m200 trajectories.
    Data must be given s.t. time t increases and starts at t=0.

    Notes on Inputs:

    t = times corresponding to given vmaxes, in Gyr

    dt = time between timesteps, in Gyr.  Expected that len(dt) = len(t)-1,
        but must be equal if len(t)==1.

    z = redshifts corresponding to t.  Must have len(t) == len(z).

    vmax = smoothed vmax trajectory of given halo in km/s, at times given by t.
        Expects len(vmax) == len(t).

    m200 = halo mass trajectory of given halo in MSUN, at times given by t.
        Expects len(m200) == len(t).

    mapping = how to map vmaxes onto SFRs

       'all' sim points
       '2bins' pre/post quench, with pre-SFR from <vmax(z>zq)>, post-SFR from vmax(z=0)
       '3bins' which adds SFR = 0 phase after reionization quenching while vmax < vthres

    scatter = True adds a lognormal scatter to SFRs.  Before reionization quenching,
        the 1-sigma symmetric scatter is 0.4 dex.

    postquench_scatter = what scatter to assume after reionization quenching

        'increasing' = adopts a scatter that's small for halos with large vmax,
            and increases towards smaller vmax
        'flat' = adopts a 1-sigma symmertic scatter of 0.3 dex
    """

    # compute threshold vmax required for star formation
    vSF = vthres if vthres != 'falling' else vmax_rejuvenation(t,z,zq=zq)

    # compute average vmax before reionization quenching
    if z[0] < zq: vavg_pre = 0.
    else:
        if len(t)==1: vavg_pre = vmax[0]
        iq = where(z>=zq)[0][-1]
        if t[iq]-t[0]==0:
            vavg_pre = vmax[iq]
        else:
            vavg_pre = sum(vmax[:iq]*dt[:iq])/(t[iq]-t[0])

    vsmooth_cored = vmax.copy()

    # compute SFHs
    if   mapping == 'all':
        sfrs = array([ sfr_pre(vv,method=prequench)       if zz > zq else \
                       (sfr_post(vv,method=postquench) if vv > v0 else 0) for vv,v0,zz in zip(vmax,vSF,z) ] )
        
    elif mapping == '2bins':
        sfrs = array([ sfr_pre(vavg_pre,method=prequench) if zz > zq else \
                       sfr_post(vmax[-1],method=postquench) for vv,zz in zip(vmax,z) ])

    elif mapping == '3bins':

        sfrs = np.zeros(len(z))
        sfrs[z>zq] = sfr_pre(vavg_pre,method=prequench)*sfr_scatter(z[z>zq],vmax[z>zq],zq=zq,prequench=prequench)

        # compute additional vmax suppression factor due to baryons
        if DMO and z[0]>zq:

            m200_pre = m200[iq]
            mstar_insitu_pre = np.sum(sfrs[:iq]*dt[:iq]*1e9)

            if m200_pre==0 or mstar_insitu_pre==0: 
                suppression = 1
            else: 
                suppression = 0.9123 * (mstar_insitu_pre/m200_pre)**-0.00479
                if suppression > 1: suppression = 1
 
            vsmooth_cored[z<=zq] = vsmooth_cored[z<=zq]*suppression

        iSFpost = (z<=zq)*(vsmooth_cored>vSF)
        sfrs[ iSFpost ] = sfr_post(vsmooth_cored[-1],method=postquench)*sfr_scatter(z[iSFpost],vmax[iSFpost],zq=zq,postquench=postquench_scatter)

        return sfrs, vsmooth_cored

    else:
        raise ValueError('SFR mapping method '+mapping+' unrecognized')

    if not scatter: return sfrs, vsmooth_cored
    else:
        return sfrs * sfr_scatter(z,vmax,zq=zq,prequench=prequench,postquench=postquench_scatter), vsmooth_cored


#################################################

def check_nsc(qmerge, zmerge, hmerge, vsmooth, zz, tt, halo, vthres=26.3, zq=4., nsc_ratio=0.7, t_delay=1):

    if len(zmerge)==0: return 0

    #finding values of all progenitors
    pro_t, pro_z, pro_r200c, pro_m200c = halo.calculate_for_progenitors('t()','z()', 'r200c','M200c')

    t_reion = np.interp(zq, pro_z, pro_t) # time when reionization occurs
    tmerge = np.interp(zmerge, pro_z, pro_t)
    imerge = [np.argmin(np.abs(np.array(pro_z) - z)) for z in zmerge]

    #dynamical timescale calculation
    tdyn = [np.sqrt(r**3/G/m) for r,m in zip(pro_r200c[imerge], pro_m200c[imerge])]
 
    #Chandraskhar dynamical friction timescale - how long to merge. 2x as takes time to relax after merging.
    merging_time = np.array([2*(0.216*q**1.3/np.log(1+q) * np.exp(1.9))*t for q,t in zip(qmerge, tdyn)])
    t_coalescence = tmerge + merging_time

    #finding major mergers after reionsation + gas buildup delay + enough time to form a NSC
    major_mergers = (tmerge > t_reion+t_delay) & (np.array(qmerge)<1/nsc_ratio) & (t_coalescence < 13.8)
    if sum(major_mergers)==0: return 0

    #finding when mergers happen so can use vsmooth
    zz_merge = [np.argmin(np.abs(np.array(zz) - z)) for z in zmerge[major_mergers]]
    zz_final = [np.argmin(np.abs(np.array(pro_t) - t)) for t in tmerge[major_mergers]]

    #record mergers which cause vmax to surpass SF threshold
    nsc_mergers=[]
    vSF = vthres*np.ones(len(tt)) if vthres != 'falling' else vmax_rejuvenation(tt,zz,zq=zq)
    for k, j in zip(zz_merge, zz_final):
        #if vsmooth[k-10 if k-10>0 else 0]<vthres and vsmooth[j]>=vthres:
        ipremerger = k-10 if k-10>0 else 0
        ipostmerger = k+10 if k+10<len(vsmooth) else len(vsmooth)-1
        if vsmooth[ipremerger] < vSF[ipremerger] and vsmooth[ipostmerger] >= vSF[ipostmerger]:
            nsc_mergers.append(k)
            print('merger z=',zz[k],'[',k,'] main halo', halo)#,'merging halo',hmerge[i][1])
            
    return 1 if len(nsc_mergers)>0 else 0 # len(nsc_mergers)


##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26.3, zq=4., plot_mergers=False, verbose=False, n=1,
                   prequench='fiducial',postquench='schechter',postquench_scatter='increasing',
                   mapping='3bins', timesteps='sim',occupation=2.5e7, DMO=False, nsc_ratio=0.7, t_delay=1):

    """
    Returns redshift, major/minor mass ratio, halo objects, and stellar mass accreted 
    for each of the given halo's mergers.  Does not compute the stellar contribution
    of mergers of mergers.

    timestep = in Gyr, or 'sims' to use simulation output timesteps
    """

    t,z,rbins,menc = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
    
    if plot_mergers:
        implot = 0
        vmax = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(menc,rbins) ]) * (sqrt(1-FBARYON) if DMO else 1)
        fig, ax = plt.subplots()
        plt.plot(t,vmax,color='k')
  
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(halo)
    msmerge = zeros((len(zmerge),n))
    nNSCmerge = zeros(len(zmerge))
  
    # record main branch components
    halos = {}
    depth = -1
    h = halo
    while h != None:
        depth += 1
        halos[ h.path ] = [ '0.'+str(depth) ]
        h = h.previous

    for ii,im in enumerate(range(len(zmerge))):
        
        for hsub in hmerge[im][1:]:

            # catch when merger tree loops back on itself --> double-counting
            depth = -1
            isRepeat = False
            h = hsub
            while h != None:
                depth += 1
                if h.path not in halos.keys():
                    halos[h.path] = [ str(im)+'.'+str(depth) ]
                else:
                    if verbose: print('--> Found repeat!!',h.path,'while tracing merger',str(int(im))+'.'+str(depth),'(also in merger(s)',halos[h.path],')')
                    halos[h.path] += [ str(im)+'.'+str(depth) ]
                    isRepeat = True
                    break
                h = h.previous
            if isRepeat: continue  # found a repeat! skip this halo

        
            # went through all fail conditions, now calculate vmax trajectory, SFH --> M*
            t_sub,z_sub,vsmooth_cored_sub,sfh_sub,mstar_insitu_sub,mstar_tot_sub,nNSC_sub = DarkLight(hsub, n=n, vthres=vthres, zq=zq,
                                                                                       prequench=prequench, postquench=postquench,
                                                                                       postquench_scatter=postquench_scatter,
                                                                                       occupation=occupation, mergers=True,
                                                                                       mapping=mapping, timesteps=timesteps, DMO=DMO,
                                                                                       nsc_ratio=nsc_ratio, t_delay=t_delay)
            if len(mstar_tot_sub[0])!=0:  
                msmerge[im] = mstar_tot_sub[:,-1]
                nNSCmerge[im]=nNSC_sub

            if plot_mergers and implot < 10:
                plt.plot(t_sub,vmax_sub,color='C'+str(im),alpha=0.25)
                plt.plot(tt_sub, vv_sub,color='C'+str(im))
                plt.plot( interp(zmerge[im],z,t), interp(zmerge[im],z,vmax) ,marker='.',color='0.7',linewidth=0)
                implot += 1

    if plot_mergers:
        plt.yscale('log')
        plt.xlabel('t (Gyr)')
        plt.ylabel(r'v$_{\rm max}$ (km/s)')
        figfn = 'mergers.pdf'
        plt.savefig(figfn)
        print('wrote',figfn)
        plt.clf()

    return zmerge, qmerge, hmerge, msmerge, nNSCmerge
