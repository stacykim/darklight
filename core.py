# darklight.py
# created 2020.03.20 by stacy kim

from numpy import *
from numpy.random import normal,random
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


def DarkLight(halo,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',
              binning='3bins',timesteps='sim',mergers=True,DMO=False,occupation=2.5e7,fn_vmax=None):
    """
    Generates a star formation history, which is integrated to obtain the M* for
    a given halo. The vmax trajectory is smoothed before applying a SFH-vmax 
    relation to reduce temporary jumps in vmax due to mergers. Returns the
    timesteps t, z, the smoothed vmax trajectory, the in-situ star formation 
    history, and M*.  If nscatter != 0, the SFH and M* are arrays with of the
    [-2simga, median, +2sigma] values.

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


    if fn_vmax==None:
        t,z,rbins,menc_dm = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
        if len(t)==0: return np.array([[]]*6)
        vmax = array([ sqrt(max( G*menc_dm[i]/rbins[i] )) for i in range(len(t)) ]) * (sqrt(1-FBARYON) if DMO else 1)
    else:
        z = halo.calculate_for_progenitors('z()')[0]
        #t,vmax = loadtxt(fn_vmax,unpack=True,usecols=(0,2))
        #t,vmax = t[::-1],vmax[::-1] # expects them to be in backwards time order
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
    
    if nscatter==0:

        if mergers != 'only':

            sfh_binned = sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=False,
                             pre_method=pre_method,post_method=post_method,post_scatter_method=post_scatter_method)
            mstar_binned = array([0] + [ sum(sfh_binned[:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ])
            if mergers == False:  return tt,zz,vsmooth,sfh_binned,mstar_binned,mstar_binned

        else:
            mstar_binned = zeros(len(tt))

        zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timesteps=timesteps,occupation=occupation,DMO=DMO,
                                                         binning=binning,nscatter=0,pre_method=pre_method,post_method=post_method,
                                                         post_scatter_method=post_scatter_method)

        mstar_tot = array([ interp(za,zz[::-1],mstar_binned[::-1]) + sum(msmerge[zmerge>=za])  for za in zz ])

        return tt,zz,vsmooth,sfh_binned,mstar_binned,mstar_tot

    else:

        sfh_binned = []
        mstar_binned = []
        mstar_binned_tot = []

        if mergers != False:
            zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timesteps=timesteps,occupation=occupation,DMO=DMO,
                                                             binning=binning,nscatter=nscatter,pre_method=pre_method,post_method=post_method,
                                                             post_scatter_method=post_scatter_method)

        for iis in range(nscatter):

            if mergers != 'only':
                sfh_binned += [ sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=True,
                                    pre_method=pre_method,post_method=post_method,post_scatter_method=post_scatter_method) ]
                mstar_binned += [ array([0] + [ sum(sfh_binned[-1][:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ]) ]
            else:
                sfh_binned += [ zeros(len(tt)) ]
                mstar_binned += [ zeros(len(tt)) ]
                
            if mergers == False:
                mstar_binned_tot = mstar_binned
            else:
                mstar_binned_tot += [ [ interp(za,zz[::-1],mstar_binned[-1][::-1]) + sum(msmerge[zmerge>=za,iis])  for za in zz ] ]

        sfh_binned = array(sfh_binned)
        mstar_binned = array(mstar_binned)
        mstar_binned_tot = array(mstar_binned_tot)
            
        return tt,zz,vsmooth,sfh_binned,mstar_binned,mstar_binned_tot if mergers==True else mstar_binned #mstar_stats  # for SFH and mstar, give [-2s,median,+2s]



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
        return np.array([ 10**normal(0,log10s) for log10s in log10scatter ])
    else:
        return array([ 10**normal(0,0.4 if zz > zre else 0.3) for zz in z ])
    
    
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
        #return array([ sfr * 10**normal(0,0.4 if zz > zre else 0.3) for sfr,zz in zip(sfrs,z) ])
        return sfrs * sfr_scatter(z,vmax,zre=zre,pre_method=pre_method,post_method=post_scatter_method)


    
##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26.3, zre=4., plot_mergers=False, verbose=False, nscatter=0,
                   pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',
                   binning='3bins', timesteps='sim',occupation=2.5e7, DMO=False):
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
    msmerge = zeros(len(zmerge)) if nscatter == 0 else zeros((len(zmerge),nscatter))
  
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

            t_sub,z_sub,rbins_sub,mencDM_sub = hsub.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
            vmax_sub = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(mencDM_sub,rbins_sub) ]) * (sqrt(1-FBARYON) if DMO else 1)

            if len(t_sub)==0:
                
                try: hsub['M200c_stars']
                except KeyError:  continue
                
                if hsub['M200c_stars'] != 0:
                    print('no mass profile data but has stars (',hsub['M200c_stars'],'msun ) for halo',hsub)

                continue  # skip if no mass profile data

            tre = interp(zre, z_sub, t_sub)

            # get subhalo's mass in case it's needed for occupation fraction below
            try: m_sub = hsub['M200c']
            except KeyError: m_sub = 1. # if no halo mass in tangos, then probably very low mass; give arbitrarily low value

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
            if len(t_sub)==1:
                zz_sub,tt_sub,vv_sub = z_sub,t_sub,vmax_sub
            elif timesteps == 'sim':
                # smooth with 500 myr timestep
                tv = arange(t_sub[-1],t_sub[0],0.5)
                vi = interp(tv,t_sub[::-1],vmax_sub[::-1])
                fv = filters.gaussian_filter(vi,sigma=1)
                if z_sub[-1] > zre:
                    ire = where(z_sub>=zre)[0][0]
                    zz_sub = concatenate([z_sub[:ire],[zre],z_sub[ire:]])[::-1]
                    tt_sub = concatenate([t_sub[:ire],[tre],t_sub[ire:]])[::-1]
                    vv_sub = interp(tt_sub, tv, fv) #concatenate([vmax_sub[:ire],[interp(zre,z_sub,vmax_sub)],vmax_sub[ire:]])[::-1] # interp in z, which approx vmax evol better
                else:
                    zz_sub,tt_sub,vv_sub = z_sub[::-1],t_sub[::-1],interp(t_sub[::-1], tv, fv) #vmax_sub[::-1]
            else:
                # smooth with given timestep
                tv = arange(t_sub[-1],t_sub[0],timesteps)
                vi = interp(tv,t_sub[::-1],vmax_sub[::-1])
                fv = filters.gaussian_filter1d(vi,sigma=1)
                # calculate usual values
                tt_sub = arange(t_sub[-1],t_sub[0],timesteps)
                if len(tt_sub)==0:
                    print('Got zero timepoints to calculate SFR for:')
                    print('t_sub',t_sub)
                    print('tt_sub',tt_sub)
                    exit()
                zz_sub = interp(tt_sub, t_sub[::-1], z_sub[::-1])
                if zz_sub[-1] > zre and interp( tt_sub[-1]+timesteps, t_sub, z_sub ) < zre:
                    append(zz_sub,[zre])
                    append(tt_sub,interp(zre,z,t))
                    print('zz_sub',zz_sub)
                elif zz_sub[-1] < zre and zre not in zz_sub:
                    izzre = where(zz_sub<zre)[0][0]
                    insert(zz_sub, izzre, zre)
                    insert(tt_sub, izzre, interp(zre,z,t))
                vv_sub = interp(tt_sub, tv, fv)
                #vv_sub = interp(tt_sub, t_sub[::-1], vmax_sub[::-1]) # for some reason no smoothing was selected - 2020.01.15
            
            #vv_sub = array([ max(vv_sub[:i+1]) for i in range(len(vv_sub)) ])  # vmaxes fall before infall, so use max vmax (after smoothing)
            if len(tt_sub)==1:
                dt_sub = array([timesteps if timesteps != 'sim' else 0.150 ]) # time resolution of EDGE
            else:
                dt_sub = tt_sub[1:]-tt_sub[:-1] # len(dt_sub) = len(tt_sub)-1
        
            pocc = occupation_fraction(vv_sub[-1],m_sub,method=occupation) # and zz_sub[-1]>=4:  #interp(vv_sub[-1], vocc, focc)


            ############################################################
            # now calculate the SFH and M* of the accreted things
            
            if nscatter == 0:

                if random() > pocc: # and zz_sub[-1]>=4:
                    msmerge[im] = 0
                else:
                    #t,z,v,sfh_binned_sub,mstar_insitu_sub,mstar_tot_sub = DarkLight(hsub,nscatter=0,vthres=vthres,zre=zre,pre_method=pre_method,
                    #                                                                post_method=post_method,post_scatter_method=post_scatter_method,
                    #                                                                binning=binning,timesteps=timesteps,DMO=DMO,occupation=occupation)
                    #msmerge[im] = mstar_tot_sub[-1]
                    sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,
                                         pre_method=pre_method,post_method=post_method,
                                         scatter=False,post_scatter_method=post_scatter_method)
                    mstar_binned_sub = array( [0] + [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(dt_sub)) ] ) # sfh_binned_sub
                    msmerge[im] = mstar_binned_sub[-1]

            else:

                for iis in range(nscatter):

                    if random() > pocc: # and zz_sub[-1]>=4:
                        msmerge[im,iis] = 0
                    else:
                        #t,z,v,sfh_binned_sub,mstar_insitu_sub,mstar_tot_sub = DarkLight(hsub,nscatter=0,vthres=vthres,zre=zre,pre_method=pre_method,
                        #                                                            post_method=post_method,post_scatter_method=post_scatter_method,
                        #                                                            binning=binning,timesteps=timesteps,DMO=DMO,occupation=occupation)
                        #msmerge[im,iis] = mstar_tot_sub[-1]
                        sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,
                                             pre_method=pre_method,post_method=post_method,
                                             scatter=True,post_scatter_method='increasing')
                        mstar_binned_sub = array( [0] + [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(dt_sub)) ] ) # sfh_binned_sub
                        msmerge[im,iis] = mstar_binned_sub[-1]

                    
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

    return zmerge, qmerge, hmerge, msmerge
