# darklight.py
# created 2020.03.20 by stacy kim

from numpy import *
from numpy.random import normal,random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

from tangos.examples.mergers import *
from .constants import *




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


def DarkLight(halo,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',
              binning='3bins',timesteps='sim',mergers=True,DMO=False,poccupied='edge1',fn_vmax=None):
    """
    Generates a star formation history, which is integrated to obtain the M* for
    a given halo. The vmax trajectory is smoothed before applying a SFH-vmax 
    relation to reduce temporary jumps in vmax due to mergers. Returns the
    timesteps t, z, the smoothed vmax trajectory, the in-situ star formation 
    history, and M*.  If nscatter != 0, the SFH and M* are arrays with of the
    [-2simga, median, +2sigma] values.

    Notes on Inputs: 

    timsteps = resolution of SFH, in Gyr, or 'sim' to use simulation timesteps.
        Used for both main and accreted halos.
    
    mergers = whether or not to include the contribution to M* from in-situ
        star formation, mergers, or both.

        True = include  M* of halos that merged with main halo
        False = only compute M* of stars that formed in-situ in main-halo
        'only' = only compute M* of mergers

    DMO = True if running on a DMO simluation.  Will then multiply particle
        masses by sqrt(1-fbary).

    poccupied = the occupation fraction to assume, if desired.
    
        'all' = all halos occupied
        'edge1' = occupation fraction derived from fiducial EDGE1 simulations
        'edge1rt' = occupation fraction from EDGE1 simulations with radiative 
            transfer; this is significantly higher than 'edge1'
        'nadler18' = from Nadler+ 2018's fit to the MW dwarfs. Note that this
            was parameterized in M200c, but we have made some simplifying 
            assumptions to convert it to vmax.
    """

    assert (mergers=='only' or mergers==True or mergers==False), "DarkLight: keyword 'mergers' must be True, False, or 'only'! Got "+str(mergers)+'.'


    if fn_vmax==None:
        t,z,rbins,menc_dm = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
        if len(t)==0: return np.array([[]]*6)
        vmax = array([ sqrt(max( G*menc_dm[i]/rbins[i] )) for i in range(len(t)) ]) * (sqrt(1-FBARYON) if DMO else 1)
    else:
        z = halo.calculate_for_progenitors('z()')[0]
        t,vmax = loadtxt(fn_vmax,unpack=True,usecols=(0,2))
        t,vmax = t[::-1],vmax[::-1] # expects them to be in backwards time order
        #t,z,vmax = loadtxt(fn_vmax,unpack=True)
        #t,z,vmax = t[::-1],z[::-1],vmax[::-1] # expects them to be in backwards time order

    
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

            sfh_binned = sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=False,pre_method=pre_method,post_method=post_method)
            mstar_binned = array([0] + [ sum(sfh_binned[:i+1]*1e9*dt[:i+1]) for i in range(len(dt)) ])
            if mergers == False:  return tt,zz,vsmooth,sfh_binned,mstar_binned,zeros(len(mstar_binned))

        else:
            mstar_binned = zeros(len(tt))

        zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timesteps=timesteps,poccupied=poccupied,DMO=DMO,
                                                         binning=binning,nscatter=0,pre_method=pre_method,post_method=post_method)

        mstar_tot = array([ interp(za,zz[::-1],mstar_binned[::-1]) + sum(msmerge[zmerge>=za])  for za in zz ])

        return tt,zz,vsmooth,sfh_binned,mstar_binned,mstar_tot

    else:

        sfh_binned = []
        mstar_binned = []
        mstar_binned_tot = []

        if mergers != False:
            zmerge, qmerge, hmerge, msmerge = accreted_stars(halo,vthres=vthres,zre=zre,timesteps=timesteps,poccupied=poccupied,DMO=DMO,
                                                             binning=binning,nscatter=nscatter,pre_method=pre_method,post_method=post_method)

        for iis in range(nscatter):

            if mergers != 'only':
                sfh_binned += [ sfh(tt,dt,zz,vsmooth,vthres=vthres,zre=zre,binning=binning,scatter=True,pre_method=pre_method,post_method=post_method) ]
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
            
        return tt,zz,vsmooth,sfh_binned,mstar_binned,mstar_binned_tot if mergers==True else mstar_stats  # for SFH and mstar, give [-2s,median,+2s]



##################################################
# OCCUPATION FRACTION

def occupation_fraction(vmax,method='edge1'):

    if method=='all':
        return 1.
    elif method=='edge1':
        vocc_edges = array([2., 9., 16., 23., 30.])
        vocc = sqrt(vocc_edges[1:]*vocc_edges[:-1])
        focc = array([0.033, 0.141, 0.250, 1.0])
        return np.interp(vmax, vocc, focc)
    elif method=='edge1rt':
        vocc_edges = array([2., 9, 16., 23., 30.])
        vocc = sqrt(vocc_edges[1:]*vocc_edges[:-1])
        focc = array([0.484, 0.591, 0.581, 1.0])
        return np.interp(vmax, vocc, focc)
    elif method=='nadler18':
        raise ValueError('occupation fraction method nadler18 not yet implemented!')
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
        
    if   method == 'fiducial' :  return 2e-7*(v/5)**3.75 * exp(v/5)  # with turn over at small vmax, SFR vmax calculated from halo birth, fit by eye
    elif method == 'smalls'   :  return 1e-7*(v/5)**4 * exp(v/5)     # with turn over at small vmax, fit by eye
    elif method == 'tSFzre4'  :  return 10**(7.66*log10(v)-12.95) # also method=='tSFzre4';  same as below, but from tSFstart to reionization (zre = 4)
    elif method == 'tSFonly'  :  return 10**(6.95*log10(v)-11.6)  # w/my SFR and vmax (max(GM/r), time avg, no forcing (0,0), no extrap), from tSFstart to tSFstop
    elif method == 'maxfilter':  return 10**(5.23*log10(v)-10.2)  # using EDGE orig + GMOs w/maxfilt vmax, SFR from 0,t(zre)
    else:  raise ValueError('Do not recognize sfr_pre method '+method)


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
        raise ValueError('SFR binning method '+binning+' unrecognized')

    if not scatter: return sfrs
    else:
        return array([ sfr * 10**normal(0,0.4 if zz > zre else 0.3) for sfr,zz in zip(sfrs,z) ])


    
##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26., zre=4., plot_mergers=False, verbose=False, nscatter=0, pre_method='fiducial',post_method='schechter',
                   binning='3bins', timesteps='sim',poccupied=True, DMO=False):
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
        t_sub,z_sub,rbins_sub,mencDM_sub = hmerge[im][1].calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
        vmax_sub = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(mencDM_sub,rbins_sub) ]) * (sqrt(1-FBARYON) if DMO else 1)
        
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
        
        pocc = occupation_fraction(vv_sub[-1],method=poccupied) # and zz_sub[-1]>=4:  #interp(vv_sub[-1], vocc, focc)


        ############################################################
        # now calculate the SFH and M* of the accreted things
            
        if nscatter == 0:

            if random() > pocc: # and zz_sub[-1]>=4:
                msmerge[im] = 0
            else:
                sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,pre_method=pre_method,post_method=post_method,scatter=False)
                mstar_binned_sub = array( [0] + [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(dt_sub)) ] ) # sfh_binned_sub
                msmerge[im] = mstar_binned_sub[-1]
                #mstar_main = interp(zmerge[im],zz[::-1],mstar_binned[::-1])
                #print('merger',im,'at z = {0:4.2f}'.format(zmerge[im]),'with {0:5.1e}'.format(mstar_binned_merge[im]),'msun stars vs {0:5.1e}'.format(mstar_main),'msun MAIN =',int(100.*mstar_binned_merge[im]/mstar_main),'%')
        else:

            for iis in range(nscatter):

                if random() > pocc: # and zz_sub[-1]>=4:
                    msmerge[im,iis] = 0
                else:
                    sfh_binned_sub = sfh(tt_sub,dt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning,scatter=True,pre_method=pre_method,post_method=post_method)
                    mstar_binned_sub = array( [0] + [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(dt_sub)) ] ) # sfh_binned_sub
                    msmerge[im,iis] = mstar_binned_sub[-1]

            #print('for merger',ii,'at',round(interp(zmerge[ii],z,t),2),'Gyr has mass',mencDM_sub[-1][-1]/1e6,'1e6 msun, vmax',round(vmax_sub[-1]),'and pocc',round(pocc,3),'. Had non-zero M*',len(nonzero(msmerge[im])[0]),'times of',nscatter)


                    
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
