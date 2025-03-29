
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



def DarkLight(halo,nscatter=1,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',
              binning='3bins',timesteps='sim',mergers=True,DMO=False,occupation='all',fn_vmax=None, nsc_ratio=0.7, t_delay=1):

    """
    Generates a star formation history, which is integrated to obtain the M* for
    a given halo. The vmax trajectory is smoothed before applying a SFH-vmax 
    relation to reduce temporary jumps in vmax due to mergers. Returns the
    timesteps t, z, the smoothed vmax trajectory, the in-situ star formation 
    history, and M*.

    Notes on Inputs: 

    timesteps = resolution of SFH, in Gyr, or 'sim' to use simulation timesteps.
        Used for both main and accreted halos.
    
    mergers = whether or not to include the contribution to M* from in-situ
        star formation, mergers, or both.

        True = include  M* of halos that merged with main halo
        False = only compute M* of stars that formed in-situ in main-halo
        'only' = only compute M* of mergers

    DMO = True if running on a DMO simluation.  Will then multiply particle
        masses by sqrt(1-fbary).

    occupation = how to determine which halos have galaxies, given as a halo mass
        in solar masses.  Less massive halos do not have galaxies, while more
        massive ones do.  By default, assumes 2.5e7 msun.  Alternatively, one 
        can adopt occupation functions from simulations or fits to data.  
        Supported ones are:

        'all' = all halos DarkLight estimates have non-zero stellar mass is occupied
        'edge1' = occupation fraction derived from fiducial EDGE1 simulations
        'edge1rt' = occupation fraction from EDGE1 simulations with radiative 
            transfer; this is significantly higher than 'edge1'
        'nadler20' = from Nadler+ 2020's fit to the MW dwarfs. Note that this
            was parameterized in M200c, but we have made some simplifying 
            assumptions to convert it to vmax.

    post_scatter_method = what scatter to assume after reionization

        'increasing' = adopts a scatter that's small for halos with large vmax,
            and increases towards smaller vmax
        'flat' = adopts a 1-sigma symmertic scatter of 0.3 dex
    """

    assert (mergers=='only' or mergers==True or mergers==False), "DarkLight: keyword 'mergers' must be True, False, or 'only'! Got "+str(mergers)+'.'

    # compute or read in vmax trajectory
    if fn_vmax==None:

        t,z,rbins,menc_dm, m200c, r200c = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile', 'M200c', 'r200c')

        if len(t)==0: 
            return np.array([]),np.array([]),np.array([]), np.array([[]]*nscatter),np.array([[]]*nscatter),np.array([[]]*nscatter), np.array([])

        vmax = np.zeros(len(t))
        for i in range(len(t)):
            vcirc = np.sqrt( G*menc_dm[i]/rbins[i] )
            vmax[i] = max(vcirc[ rbins[i]<r200c[i] ])  # make sure rmax < r200

    else:
        if fn_vmax == '../outliers/vmax-pynbody_halo600lm.dat':
            z = halo.calculate_for_progenitors('z()')[0]
            t,vmax = loadtxt(fn_vmax,unpack=True,usecols=(0,2))
            t,vmax = t[::-1],vmax[::-1] # expects them to be in backwards time order
        else:
            t,z,vmax = loadtxt(fn_vmax,unpack=True)
            t,z,vmax = t[::-1],z[::-1],vmax[::-1] # expects them to be in backwards time order

    
    ############################################################
    # Get values at points where DarkLight SFH will be calculated

    if timesteps == 'sim':
        tt = t[::-1]  # reverse arrays so time is increasing
        zz = z[::-1]
    else:
        tt = arange(t[-1],t[0],timesteps)
        zz = interp(tt, t[::-1], z[::-1])
    
    # make sure reionization included in time steps, if halo exists then
    if zz[-1] < zre and zz[0] > zre:
        if zre not in zz:
            ire = where(zz<=zre)[0][0]
            tt = concatenate([ tt[:ire], [interp(zre,z,t)], tt[ire:] ])
            zz = concatenate([ zz[:ire], [zre],             zz[ire:] ])

    dt = tt[1:]-tt[:-1] # since len(dt) = len(t)-1, need to be careful w/indexing below

    if len(t) > 1:
        vsmooth = smooth(t[::-1],vmax[::-1],tt,sigma=0.5) # smoothed over 500 Myr
    else:
        vsmooth = vmax

        
    ############################################################
    # Generate the star formation histories
    nNSC=0
    # check if halo is occupied
    m = halo['M200c'] if 'M200c' in halo.keys() else 1. # if no mass in tangos, then probably very low mass, give arbitrarily low value
    pocc = occupation_fraction(vmax[-1],m,method=occupation)
    occupied = np.random.rand(nscatter) < pocc

    # compute in-situ component
    sfhs_insitu = np.zeros((nscatter,len(tt)))
    mstars_insitu = np.zeros((nscatter,len(tt)))

    for iis in range(nscatter):
        if occupied[iis] and mergers != 'only':
            sfhs_insitu[iis]   = sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=True,
                                     pre_method=pre_method,post_method=post_method,post_scatter_method=post_scatter_method)
            mstars_insitu[iis] = np.array([0] + [ sum(sfhs_insitu[iis][:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ])


    # compute accreted component
    mstars_accreted = np.zeros((nscatter,len(tt)))
    if mergers and sum(occupied)>0:
        zmerge, qmerge, hmerge, msmerge, nNSCmerge = accreted_stars(halo,vthres=vthres,zre=zre,timesteps=timesteps,occupation='all',DMO=DMO,
                                                         binning=binning,nscatter=sum(occupied),pre_method=pre_method,post_method=post_method,
                                                         post_scatter_method=post_scatter_method, nsc_ratio=nsc_ratio, t_delay=t_delay)
        mstars_accreted[occupied] = np.array([np.sum(msmerge[zmerge>z],axis=0) for z in zz]).T  # change from mstar for each merger -> cumsum(mstar) for each time
           

        nNSC += check_nsc(qmerge, zmerge, hmerge, vsmooth, zz, halo, vthres=vthres, zre=zre, nsc_ratio=nsc_ratio, t_delay=t_delay)
        #print('nNSC from check nsc: ', nNSC)
        #print('nNSCmerge: ', nNSCmerge) #array of 0s
        nNSC += sum(nNSCmerge)
        #print('nNSC final: ', nNSC)      
            
       
    # compute total stellar mass and we're done!
    mstars_tot = mstars_insitu + mstars_accreted
    return tt,zz,vsmooth,sfhs_insitu,mstars_insitu,mstars_tot, nNSC




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

def sfr_pre(vmax,method='fiducial'):

    if not hasattr(vmax,'__iter__'):
        v = vmax if vmax<=20 else 20
    else:
        v = vmax[:]
        v[ v>20 ] = 20.
    #v = vmax

    if   method == 'fiducial': return 10**(6.78*log10(v)-11.6)  # no turn over, simple log-linear fit to dataset below
    elif method == 'fiducial_with_turnover' :  return 2e-7*(v/5)**3.75 * exp(v/5)  # with turn over at small vmax, SFR vmax calculated from halo birth, fit by eye
    elif method == 'smalls'   :  return 1e-7*(v/5)**4 * exp(v/5)     # with turn over at small vmax, fit by eye
    elif method == 'tSFzre4'  :  return 10**(7.66*log10(v)-12.95) # also method=='tSFzre4';  same as below, but from tSFstart to reionization (zre = 4)
    elif method == 'tSFonly'  :  return 10**(6.95*log10(v)-11.6)  # w/my SFR and vmax (max(GM/r), time avg, no forcing (0,0), no extrap), from tSFstart to tSFstop
    elif method == 'maxfilter':  return 10**(5.23*log10(v)-10.2)  # using EDGE orig + GMOs w/maxfilt vmax, SFR from 0,t(zre)
    else:  raise ValueError('Do not recognize sfr_pre method '+method)


def sfr_post(vmax,method='schechter'):
    if   method == 'schechter'    :  return 7.06 * (vmax/182.4)**3.07 * exp(-182.4/vmax)  # schechter fxn fit
    elif method == 'schechter_mc' :  return 0.22 * (vmax/85.)**3.71 * np.exp(-85./vmax)   # schechter b/t fiducial and mid
    elif method == 'schechter_mid':  return 0.4 * (vmax/100.)**3.71 * np.exp(-100./vmax)  # schechter fxn b/t fiducial and MW
    elif method == 'schechterMW'  :  return 6.28e-3 * (vmax/43.54)**4.35 * exp(-43.54/vmax)  # schechter fxn fit w/MW dwarfs
    elif method == 'linear'       :  return 10**( 5.48*log10(vmax) - 11.9 )  # linear fit w/MW dwarfs


def sfr_scatter(z, vmax, zre=4., pre_method='fiducial', post_method='increasing'):
    """
    Returns scatter in the SFR-vmax relation.  Assumes 0.4 dex lognormal scatter
    before reionization.  Post-reionization scatter is determined by given method.

    z and vmax must be arrays of the same length.

    Notes on Inputs:

    method = type of relation to adopt after reionization

        'increasing' = scatter increases for smaller halos (lower vmax)
        'flat'       = 0.3 dex lognormal scatter (independent of mass)
    """

    if post_method=='increasing':  # increasing scatter for small vmax post-reionization
        log10scatter = array([ 0.4 if zz > zre else (-0.651*log10(vv)+1.74) for zz,vv in zip(z,vmax) ])
        log10scatter[ log10scatter < 0.2 ] = 0.2 # max out at 0.2 dex at high-mass end, when extrapolating above fit
        return np.array([ 10**np.random.normal(0,log10s) for log10s in log10scatter ])
    else:
        return array([ 10**np.random.normal(0,0.4 if zz > zre else 0.3) for zz in z ])
    
    
def sfh(t, dt, z, vmax, vthres=26.3, zre=4.,binning='3bins',pre_method='fiducial',post_method='schechter',
        post_scatter_method='increasing',scatter=False):
    """
    Assumes we are given a halo's entire vmax trajectory.
    Data must be given s.t. time t increases and starts at t=0.
    Expected that len(dt) = len(t)-1, but must be equal if len(t)==1.

    Notes on Inputs:

    binning = how to map vmaxes onto SFRs

       'all' sim points
       '2bins' pre/post reion, with pre-SFR from <vmax(z>zre)>, post-SFR from vmax(z=0)
       '3bins' which adds SFR = 0 phase after reion while vmax < vthres

    scatter = True adds a lognormal scatter to SFRs.  Pre-reionization, the 
        1-sigma symmetric scatter is 0.4 dex.

    post_scatter_method = what scatter to assume after reionization

        'increasing' = adopts a scatter that's small for halos with large vmax,
            and increases towards smaller vmax
        'flat' = adopts a 1-sigma symmertic scatter of 0.3 dex
    """
    if z[0] < zre: vavg_pre = 0.
    else:
        if len(t)==1: vavg_pre = vmax[0]
        ire = where(z>=zre)[0][-1]
        if t[ire]-t[0]==0:
            vavg_pre = vmax[ire]
        else:
            vavg_pre = sum(vmax[:ire]*dt[:ire])/(t[ire]-t[0]) #mean([ vv for vv,zz in zip(vmax,z) if zz > zre ])
    if   binning == 'all':
        sfrs = array([ sfr_pre(vv,method=pre_method)       if zz > zre else (sfr_post(vv,method=post_method) if vv > vthres else 0) for vv,zz in zip(vmax,z) ] )
    elif binning == '2bins':
        sfrs = array([ sfr_pre(vavg_pre,method=pre_method) if zz > zre else sfr_post(vmax[-1],method=post_method) for vv,zz in zip(vmax,z) ])
    elif binning == '3bins':
        sfrs = array([ sfr_pre(vavg_pre,method=pre_method) if zz > zre else (sfr_post(vmax[-1],method=post_method) if vv > vthres else 0) for vv,zz in zip(vmax,z) ])
    else:
        raise ValueError('SFR binning method '+binning+' unrecognized')

    if not scatter: return sfrs
    else:
        #return array([ sfr * 10**np.random.normal(0,0.4 if zz > zre else 0.3) for sfr,zz in zip(sfrs,z) ])
        return sfrs * sfr_scatter(z,vmax,zre=zre,pre_method=pre_method,post_method=post_scatter_method)

#################################################

#from astropy.cosmology import Planck15 as cosmo  
#from astropy.cosmology import z_at_value
#import astropy.units as u

def check_nsc(qmerge, zmerge, hmerge, vsmooth, zz, halo, vthres=26.3, zre=4., nsc_ratio=0.7, t_delay=1):

    if len(zmerge)==0: return 0

    #finding values of all progenitors
    pro_t, pro_z, pro_r200c, pro_m200c = halo.calculate_for_progenitors('t()','z()', 'r200c','M200c')

    t_reion = np.interp(zre, pro_z, pro_t) # time when reionization occurs
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
    #for i in np.argwhere(major_mergers):
    for k, j in zip(zz_merge, zz_final):
        #if vsmooth[k-10 if k-10>0 else 0]<vthres and vsmooth[j]>=vthres:
        if vsmooth[k-10 if k-10>0 else 0]<vthres and vsmooth[k+10 if k+10<len(vsmooth) else len(vsmooth)-1]>=vthres:
            nsc_mergers.append(k)
            print('merger z=',zz[k],'[',k,'] main halo', halo)#,'merging halo',hmerge[i][1])
            
    return 1 if len(nsc_mergers)>0 else 0 # len(nsc_mergers)



##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26.3, zre=4., plot_mergers=False, verbose=False, nscatter=1,
                   pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',
                   binning='3bins', timesteps='sim',occupation=2.5e7, DMO=False, nsc_ratio=0.7, t_delay=1):
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
    msmerge = zeros((len(zmerge),nscatter))
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
            t_sub,z_sub,vsmooth_sub,sfh_sub,mstar_insitu_sub,mstar_tot_sub,nNSC_sub = DarkLight(hsub, nscatter=nscatter, vthres=vthres, zre=zre,
                                                                                       pre_method=pre_method, post_method=post_method,
                                                                                       post_scatter_method=post_scatter_method,
                                                                                       occupation=occupation, mergers=True,
                                                                                       binning=binning, timesteps=timesteps, DMO=DMO, nsc_ratio=nsc_ratio, t_delay=t_delay)
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
