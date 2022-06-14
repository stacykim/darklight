import numpy as np

def get_haloIDs_of_duplicates(timestep):

    # root out false halos that have same progenitor as another
    m200c, ih0,ih1,ih2,ih3, contamination = timestep.calculate_all('M200c','halo_number()','earlier(1).halo_number()','earlier(2).halo_number()','earlier(3).halo_number()','contamination_fraction')

    false_haloIDs = np.array([])
    
    for step in [ih1, ih2, ih3]:
        IDs,n = np.unique(ih1,return_counts=True)
        for ID in IDs[n>1]:
            dupes = ih0[np.where(ih1==ID)]
            false_haloIDs = np.append(false_haloIDs, np.delete(dupes,dupes.argmin())) # remove all except the halo with the lowest ID

    false_haloIDs = np.unique(false_haloIDs)
    print('found',len(false_haloIDs),'false halos')
    return false_haloIDs
