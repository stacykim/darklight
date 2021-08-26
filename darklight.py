# darklight.py
# created 2020.03.20 by stacy kim

from numpy import *
from numpy.random import normal
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

from tangos.examples.mergers import *

G   = 4.30092e-6  # kpc km^2 / MSUN / s^2



##################################################
# DARKLIGHT

def DarkLight(halo,nscatter=0,vthres=26.3,zre=4.,binning='3bins',pre_method='fiducial',post_method='schechter',timestepping=0.25,mergers=True):
    """
    Generates a star formation history, which is integrated to obtain the M* for
    a given halo. The vmax trajectory is smoothed before applying a SFH-vmax 
    relation to reduce temporary jumps in vmax due to mergers. Returns the
    timesteps t, z, the smoothed vmax trajectory, and M*.

    Notes on Inputs: 
    mergers = whether or not to include the contribution to M* from in-situ
        star formation, mergers, or both.

        True = include  M* of halos that merged with main halo
        False = only compute M* of stars that formed in-situ in main-halo
        'only' = only compute M* of mergers

    timestepping = resolution of SFH, in Gyr
    """

    assert (mergers=='only' or mergers==True or mergers==False), "DarkLight: keyword 'mergers' must be True, False, or 'only'! Got "+str(mergers)+'.'

    
    t,z,rbins,menc_dm = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
    ntimesteps = len(t)
    vmax = array([ sqrt(max( G*menc_dm[i]/rbins[i] )) for i in range(ntimesteps) ])
    ire = where(z>=zre)[0][0]

    # smooth vmax rotation curve
    t500myr = arange(t[-1],t[0],0.5)  # NOTE: interpolated in 500 Myr bins
    v500myr = interp(t500myr,t[::-1],vmax[::-1])
    vsmooth500myr = filters.gaussian_filter1d(v500myr,sigma=1)    

    # make sure reionization included in time steps
    if zre not in z:
        t = concatenate([ t[:ire], [interp(zre,z,t)], t[ire:] ])
        z = concatenate([ z[:ire], [zre], z[ire:] ])
    tt = t[::-1]  # reverse arrays so time is increasing
    zz = z[::-1]
    vsmooth = interp(tt, t500myr, vsmooth500myr)
    dt = tt[1:]-tt[:-1] # since len(dt) = len(t)-1, need to be careful w/indexing below
    
    # generate the star formation histories
    if nscatter==0:

        if mergers != 'only':

            sfh_binned = sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=False,pre_method=pre_method,post_method=post_method)
            mstar_binned = array([0] + [ sum(sfh_binned[:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ])
            if mergers == False:  return tt,zz,vsmooth,mstar_binned

        else:
            mstar_binned = zeros(len(tt))

        zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timestep=timestepping,
                                                         binning=binning,nscatter=0,pre_method=pre_method,post_method=post_method)

        zall = list( set(zz) | set(zmerge) )
        zall.sort(reverse=True)
        assert len(zall)==len(zz),'DarkLight: redshift of merger(s) not in simulation redshifts! DarkLight will return arrays of mismatched length'

        mstar_tot = [ interp(za,zz[::-1],mstar_binned[::-1]) + sum(msmerge[zmerge>=za])  for za in zall ]

        return tt,zz,vsmooth,mstar_tot

    else:

        sfh_binned = []
        mstar_binned = []
        mstar_binned_tot = []

        zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timestep=timestepping,
                                                         binning=binning,nscatter=nscatter,pre_method=pre_method,post_method=post_method)
        zall = list( set(zz) | set(zmerge) )
        zall.sort(reverse=True)
        assert len(zall)==len(zz),'DarkLight: redshift of merger(s) not in simulation redshifts! DarkLight will return arrays of mismatched length'

        for iis in range(nscatter):

            if mergers != 'only':
                sfh_binned += [ sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=True,pre_method=pre_method,post_method=post_method) ]
                mstar_binned += [ array([0] + [ sum(sfh_binned[-1][:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ]) ]
            else:
                sfh_binned += [ zeros(len(tt)) ]
                mstar_binned += [ zeros(len(tt)) ]
                
            mstar_binned_tot += [ [ interp(za,zz[::-1],mstar_binned[-1][::-1]) + sum(msmerge[zmerge>=za,iis])  for za in zall ] ]

        sfh_binned = array(sfh_binned)
        mstar_binned = array(mstar_binned)
        mstar_binned_tot = array(mstar_binned_tot)

        sfh_stats       = array([ percentile(sfh_binned      [:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(tt)) ])
        mstar_stats     = array([ percentile(mstar_binned    [:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(tt)) ])
        mstar_tot_stats = array([ percentile(mstar_binned_tot[:,i], [15.9,50,84.1, 2.3,97.7]) for i in range(len(zall)) ])
            
        return tt,zz,vsmooth,mstar_tot_stats[1] if mergers==True else mstar_stats[1]  # use the medians



##################################################
# SFH ROUTINES

def sfr_pre(vmax,method='fiducial'):
    if vmax > 20: vmax = 20.
    if   method == 'fiducial' :  return 2e-7*(vmax/5)**3.75 * exp(vmax/5)  # with turn over at small vmax, SFR vmax calculated from halo birth, fit by eye
    elif method == 'smalls'   :  return 1e-7*(vmax/5)**4 * exp(vmax/5)     # with turn over at small vmax, fit by eye
    elif method == 'tSFzre4'  :  return 10**(7.66*log10(vmax)-12.95) # also method=='tSFzre4';  same as below, but from tSFstart to reionization (zre = 4)
    elif method == 'tSFonly'  :  return 10**(6.95*log10(vmax)-11.6)  # w/my SFR and vmax (max(GM/r), time avg, no forcing (0,0), no extrap), from tSFstart to tSFstop
    elif method == 'maxfilter':  return 10**(5.23*log10(vmax)-10.2)  # using EDGE orig + GMOs w/maxfilt vmax, SFR from 0,t(zre)
    else:
        print('Do not recognize sfr_pre method',method,'.  Aborting...')
        exit()


def sfr_post(vmax,method='schechter'):
    if   method == 'schechter'  :  return 7.06 * (vmax/182.4)**3.07 * exp(-182.4/vmax)  # schechter fxn fit
    if   method == 'schechterMW':  return 6.28e-3 * (vmax/43.54)**4.35 * exp(-43.54/vmax)  # schechter fxn fit w/MW dwarfs
    elif method == 'linear'     :  return 10**( 5.48*log10(vmax) - 11.9 )  # linear fit w/MW dwarfs


def sfh(t, dt, z, vmax, vthres=26.3, zre=4.,binning='3bins',pre_method='fiducial',post_method='schechter',scatter=False):
    """
    Assumes we are given a halo's entire vmax trajectory.
    Data must be given s.t. time t increases and starts at t=0.
    Expected that len(dt) = len(t)-1, but must be equal if len(t)==1.

    binning options:
       'all' sim points
       '2bins' pre/post reion, with pre-SFR from <vmax(z>zre)>, post-SFR from vmax(z=0)
       '3bins' which adds SFR = 0 phase after reion while vmax < vthres

    Setting scatter==True adds a lognormal scatter to SFRs
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
        print('SFR binning method',binning,'unrecognized!  Aborting...')
        exit()

    if not scatter: return sfrs
    else:
        return array([ sfr * 10**normal(0,0.4 if zz > zre else 0.3) for sfr,zz in zip(sfrs,z) ])


    
##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26., zre=4., binning='3bins', plot_mergers=False, timestep=0.250, verbose=False, nscatter=0, pre_method='fiducial',post_method='schechter'):
    """
    Returns redshift, major/minor mass ratio, halo objects, and stellar mass accreted 
    for each of the given halo's mergers.  Does not compute the stellar contribution
    of mergers of mergers.

    timestep = in Gyr, or 'orig' to use simulation output timesteps
    """

    t,z,rbins,menc = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
    
    if plot_mergers:
        implot = 0
        vmax = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(menc,rbins) ])
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
        t_sub,z_sub,rbins_sub,mencDM_sub = hmerge[im][1].calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
        vmax_sub = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(mencDM_sub,rbins_sub) ])
        
        if len(t_sub)==0: continue  # skip if no mass profile data
        tre = interp(zre, z_sub, t_sub)
        
        # catch when merger tree loops back on itself --> double-counting
        h = hmerge[im][1]
        depth = -1
        isRepeat = False
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
        if isRepeat: continue  # found a repeat! skip this halo.
                
        # went through all fail conditions, now calculate vmax trajectory, SFH --> M*
        if len(t_sub)==1:
            zz_sub,tt_sub,vv_sub = z_sub,t_sub,vmax_sub
        elif timestep == 'orig':
            # smooth with 500 myr timestep
            tv = arange(t[-1],t[0],0.5)
            vi = interp(tv,t[::-1],vmax_sub[::-1])
            fv = filters.gaussian_filter(vi,sigma=1)
            if z_sub[0] < zre:
                ire = where(z>=zre)[0][0]
                zz_sub = concatenate([z_sub[:ire],[zre],z_sub[ire:]])[::-1]
                tt_sub = concatenate([t_sub[:ire],[tre],t_sub[ire:]])[::-1]
                vv_sub = interp(tt, tv, fv) #concatenate([vmax_sub[:ire],[interp(zre,z_sub,vmax_sub)],vmax_sub[ire:]])[::-1] # interp in z, which approx vmax evol better
            else:
                zz_sub,tt_sub,vv_sub = z_sub[::-1],t_sub[::-1],interp(tt, tv, fv) #vmax_sub[::-1]
        else:
            # smooth with given timestep
            tv = arange(t_sub[-1],t_sub[0],timestep)
            vi = interp(tv,t_sub[::-1],vmax_sub[::-1])
            fv = filters.gaussian_filter1d(vi,sigma=1)
            # calculate usual values
            tt_sub = arange(t_sub[-1],t_sub[0],timestep)
            if len(tt_sub)==0:
                print('Got zero timepoints to calculate SFR for:')
                print('t_sub',t_sub)
                print('tt_sub',tt_sub)
                exit()
            zz_sub = interp(tt_sub, t_sub[::-1], z_sub[::-1])
            if zz_sub[-1] > zre and interp( tt_sub[-1]+timestep, t_sub, z_sub ) < zre:
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
        dt_sub = array([timestep]) if len(tt_sub)==1 else tt_sub[1:]-tt_sub[:-1] # len(dt_sub) = len(tt_sub)-1
        
        if nscatter == 0:
            sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,pre_method=pre_method,post_method=post_method,scatter=False)
            mstar_binned_sub = array( [0] + [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(dt_sub)) ] ) # sfh_binned_sub
            msmerge[im] = mstar_binned_sub[-1]
            #mstar_main = interp(zmerge[im],zz[::-1],mstar_binned[::-1])
            #print('merger',im,'at z = {0:4.2f}'.format(zmerge[im]),'with {0:5.1e}'.format(mstar_binned_merge[im]),'msun stars vs {0:5.1e}'.format(mstar_main),'msun MAIN =',int(100.*mstar_binned_merge[im]/mstar_main),'%')
        else:
            for iis in range(nscatter):
                sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,scatter=True,pre_method=pre_method,post_method=post_method)
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
