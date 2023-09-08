import os, glob
import numpy as np
from ..constants import *


# sim names
edge1_sims   = [ 'Halo600_fiducial', 'Halo600_fiducial_later_mergers',
               'Halo605_fiducial',
               'Halo624_fiducial', 'Halo624_fiducial_higher_finalmass',
               'Halo1445_fiducial',
               'Halo1459_fiducial', 'Halo1459_fiducial_Mreionx02', 'Halo1459_fiducial_Mreionx03', 'Halo1459_fiducial_Mreionx12'
             ]
edge1rt_sims = ['Halo600_RT', 'Halo605_RT', 'Halo624_RT', 'Halo1445_RT', 'Halo1459_RT']
chimera_sims   = ['Halo383_fiducial', 'Halo383_fiducial_288', 'Halo383_fiducial_early', 'Halo383_fiducial_late', 'Halo383_Massive']

edge2rerun_sims = ['Halo1459_RT', 'Halo1459_Mreionx02_RT', 'Halo1459_Mreionx03_RT',
                   'Halo1459_Mreionx03_RT_hires',
                   'Halo1445_RT', 'Halo605_RT', 'Halo624_RT',
                   'Halo383_RT', 'Halo383_stochastic_RT', 'Halo383_early_RT', 'Halo383_late_RT', 'Halo383_Massive_RT']
edge2_sims = ['Halo153_RT', 'Halo153_early_RT', 'Halo153_late_RT',
              'Halo261_RT', 'Halo261_EARLY_DECON_RT', 'Halo261_LATE_DECON_RT',
              'Halo339_RT'] #, 'Halo339_early_RT', 'Halo339_late_RT']

all_edge1_sims = edge1_sims + chimera_sims + edge1rt_sims
all_edge2_sims = edge2_sims + edge2rerun_sims



def get_shortname(simname, physics='edge1'):
    """
    Returns the halo number and a shortened version of the simulation name,
    e.g. giving 'Halo600_fiducial_later_mergers' returns '600' and '600lm'.
    """

    if simname=='void_volume':  return 'ALL','void'
    
    split = simname.split('_')
    shortname = split[0][4:]
    halonum = shortname[:]

    hires = 'hires' in split
    if hires: split.remove('hires')

    if len(split) > 2:

        # EDGE2 WDMs
        if 'wdm' in simname:
            shortname += 'w'+simname.split('wdm')[1][0]
        if physics=='edge2':
            if 'early' in simname.lower():  shortname += 'e'
            elif 'late' in simname.lower(): shortname += 'l'

        # EDGE1
        elif halonum=='332': shortname += 'low'
        elif halonum=='383': shortname += 'late'
        elif halonum=='600': shortname += 'lm'
        elif halonum=='624': shortname += 'hm'
        elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
        elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
        elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
        else:  raise ValueError('unsupported simulation '+simname+'! Not sure what shortname to give it.')

    elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'

    if 'DMO' in simname:  shortname += '_dmo'
    if hires:             shortname += '_hires'

    return halonum, shortname



def get_number_of_snapshots(simname,machine='astro',physics='edge1'):
    
    path = get_pynbody_path(simname,machine=machine,physics=physics)
    snapshots = glob.glob(os.path.join(path,'output_?????'))
    return len(snapshots)



def load_tangos_data(simname,machine='astro',physics='edge1',verbose=True):
 
    if physics=='edge2':  raise ValueError('no tangos databases available yet for EDGE2 suite!')


    import tangos
   
    halonum, shortname = get_shortname(simname,physics=physics)

    if machine=='astro':

        if halonum=='153' or halonum=='261' or halonum=='339':
            raise OSError('tangos databases do not yet exist for EDGE2 simulations!')
        elif halonum=='383':
            tangos_path = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
        else:
            # need to add support for EDGE1 reruns once databases made.
            tangos_path = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'

    elif machine=='dirac':

        if halonum=='383':
            tangos_path = '/scratch/dp101/shared/EDGE/tangos/'
        elif halonum=='153' or halonum=='261' or halonum=='339':
            tangos_path = '/scratch/dp191/shared/tangos/'
        else:
            # need to add support for EDGE1 reruns once databases made.
            tangos_path = '/scratch/dp101/shared/EDGE/tangos/'

    else:
        raise ValueError('support for machine '+machine+' not implemented!')

    if verbose: print('looking for database at',tangos_path+('void.db' if simname=='void_volume' else 'Halo'+halonum+'.db'))
    

    # get the data
    if simname=='void_volume':
        tangos.core.init_db(tangos_path+'void.db')
    else:
        tangos.core.init_db(tangos_path+'Halo'+halonum+'.db')

    sim = tangos.get_simulation(simname)

    return sim



def get_pynbody_path(simname,machine='astro',physics='edge1'):

    """
    physics = 'edge1' or 'edge2'
    """

    if physics=='edge1' and simname not in all_edge1_sims:  raise ValueError(simname+' not in EDGE1 suite!')
    if physics=='edge2' and simname not in all_edge2_sims:  raise ValueError(simname+' not in EDGE2 suite!')


    halonum, shortname = get_shortname(simname, physics=physics)

    if machine=='astro':

        if physics=='edge1':

            if halonum=='383':
                return '/vol/ph/astro_data/shared/etaylor/CHIMERA/{0}/'.format(simname)
            elif halonum != 'ALL' and halonum != shortname and 'hires' not in shortname and 'RT' not in simname:
                return '/vol/ph/astro_data2/shared/morkney/EDGE_GM/{0}/'.format(simname)
            else:
                return '/vol/ph/astro_data/shared/morkney/EDGE/{0}/'.format(simname)

        elif physics=='edge2':
            raise OSError('particle data not yet available on astro servers for EDGE2 simulations!')


    elif machine == 'dirac':

        if physics == 'edge1':

            if halonum=='383':
                return '/scratch/dp191/shared/CHIMERA/{0}/'.format(simname)
            else:
                return '/scratch/dp101/shared/EDGE/{0}/'.format(simname)

        elif physics=='edge2':

            if halonum=='153' or halonum=='261' or halonum=='339':
                return '/scratch/dp191/shared/EDGE2_simulations/{0}/'.format(simname)
            else:
                return '/scratch/dp191/shared/RT_rerun_simulations/{0}/'.format(simname)


    else:
        raise ValueError('support for machine '+machine+' not implemented!')



def load_pynbody_data(simname,output=-1,machine='astro',verbose=True,physics='edge1'):
    """
    Returns the particle data for the given simulation and output number.
    By default, returns the z=0 output, which is specified via '-1'.
        suite = 'edge1' or 'edge2'  # the physics the sims were run with
    """

    if physics=='edge1' and simname not in all_edge1_sims:  raise ValueError(simname+' not in EDGE1 suite!')
    if physics=='edge2' and simname not in all_edge2_sims:  raise ValueError(simname+' not in EDGE2 suite!')


    import pynbody
    
    path = get_pynbody_path(simname,machine=machine,physics=physics)

    if not os.path.isdir(path):
        raise FileNotFoundError('Full hydro particle data does not exist! (looked in '+path+')')

    if output == -1:
        output = get_number_of_snapshots(simname,machine=machine,physics=physics)

    simfn = os.path.join(path,'output_'+str(output).zfill(5))
    particles = pynbody.load(simfn)
    if verbose:  print('read',simfn)
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

        if t_old[-1] < t_new[i]:
            sfh_new[i:] = np.zeros(len(sfh_new[i:]))
            break

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
    

