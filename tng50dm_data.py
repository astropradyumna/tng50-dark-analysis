# This is code to extract the subhalo data of Type-1 and Type-2 subhalos from the simulation for TNG50-1-Dark

import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed #This is to parallelize the code
from paths_for_files import * #This is to get the paths for the files
import logging
import sys

h = 0.6774 # Hubble constant / 100 km/s/Mpc
mdm = 5.4e5 # This is the dark matter particle mass in solar masses in the TNG50-1-Dark simulation.


ages_df = pd.read_csv('ages_tng.csv', comment = '#') # This is a file that contains redhsifts and ages of universe at different snapshots

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])

#Let fofno be the input to the code from the terminal
fofno = int(os.sys.argv[1])  #This has to be an integer
this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fofno)
central_id_at99 = this_fof['GroupFirstSub']
this_fof_nsubs = this_fof['GroupNsubs']


central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'SubhaloVel', 'SubhaloCM', 'Group_M_Crit200'] 
central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = central_fields, onlyMPB = True)
central_snaps = central_tree['SnapNum']
central_redshift = all_redshifts[central_snaps]
central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
central_vx = central_tree['SubhaloVel'][:, 0]
central_vy = central_tree['SubhaloVel'][:, 1]
central_vz = central_tree['SubhaloVel'][:, 2]
Rvir = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii



def get_grnr(snap):
    '''
    This function returns the group number at a given snapshot 
    snap: input snapshot
    '''
    grnr_arr = central_tree['SubhaloGrNr']
    grnr = grnr_arr[np.where(central_snaps == snap)]
    return grnr


central_grnr = np.zeros(0)

for csnap in central_snaps:
    central_grnr = np.append(central_grnr, get_grnr(csnap))



# We will have to load the particle data as well for this FoF




def get_t1subhalo_data(sfid):  
    '''
    This is for Type-1 in the simulation
    '''
    this_subh = il.groupcat.loadSingle(basePath, 99, subhaloID = sfid)
    pos = np.array(this_subh['SubhaloPos'])/h - (central_x[0], central_y[0], central_z[0])
    cm = np.array(this_subh['SubhaloCM'])/h - (central_x[0], central_y[0], central_z[0])

    vel = np.array(this_subh['SubhaloVel'] - (central_vx[0], central_vy[0], central_vz[0]))
    
    
    pos_ar = pos.reshape(1, -1)
    cm_ar = cm.reshape(1, -1)
    vel_ar = vel.reshape(1, -1)

    # Let us now get the vmax_if by looking at the merger tree
    if_tree = il.sublink.loadTree(basePath, int(99), int(sfid),
            fields = ['SubfindID', 'SnapNum', 'SubhaloMass', 'SubhaloGrNr', 'GroupFirstSub', 'SubhaloVmax', 'SubhaloPos'], 
            onlyMPB = True) #Progenitor tree
    if_snap = if_tree['SnapNum']
    snap_len = len(if_snap)
    if_grnr = if_tree['SubhaloGrNr']
    # ixs_for_central = np.where(np.isin(central_snaps, if_snap))[0]
    # central_pos = np.column_stack((central_x[ixs_for_central],
    #                                 central_y[ixs_for_central],
    #                                 central_z[ixs_for_central]))
    
    # subh_x = if_tree['SubhaloPos'][:, 0]/h/(1 + all_redshifts[if_snap]) - central_x[ixs_for_central]
    # subh_y = if_tree['SubhaloPos'][:, 1]/h/(1 + all_redshifts[if_snap]) - central_y[ixs_for_central]
    # subh_z = if_tree['SubhaloPos'][:, 2]/h/(1 + all_redshifts[if_snap]) - central_z[ixs_for_central]

    # if_dist = np.sqrt(subh_x**2 + subh_y**2 + subh_z**2)


    vpeak = np.max(if_tree['SubhaloVmax']) #This is the peak value of Vmax for the subhalo across all snapshots

    i1 = 0
    i2 = 0
    inf1_snap = -1
    inf1_sfid = -1
    matching_snap = -1

    i3 = 0
    crossing_snap = -1
    crossing_sfid = -1


    for ix in range(len(if_snap)):
        '''
        This loop is to go through all the snaps in order to obtain the snap where infall happened
        '''
        snap_ix = if_snap[snap_len - ix - 1] #Go in an ascending order of snapshots for this variable
        
        # print(if_snap, if_grnr, central_grnr)
        
        if (i1 == 0) and (if_tree['SubfindID'][ix] == if_tree['GroupFirstSub'][ix]):
            inf1_snap = if_snap[ix] #This would be the last time when it made transition from central to a satellite
            i1 = 1
            # print(snap, subid, inf1_snap)
        # if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
        if if_grnr[snap_len - ix - 1].size * central_grnr[central_snaps == snap_ix].size > 0: #What is this for? Assuming this is a check if subhalo existed at this snapshot
            if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
                matching_snap = snap_ix #This would be the first time when this entered FoF halo 0
                i2 = 1
        # if (i3 == 0) and (if_dist[snap_len - ix - 1] < Rvir[central_snaps == snap_ix]):
        #     crossing_snap = snap_ix
        #     crossing_sfid = if_tree['SubfindID'][snap_len - ix - 1]
        #     i3 = 1
        
    
        if i1*i2 == 1:
            # print(pos_ar[:, 0][0]) 
            # column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'cmx_ar', 'cmy_ar', 'cmz_ar', 'vmax_ar', 'mass_ar', 'len_ar', 'sfid_ar', 
                # 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'mdm_if_ar', 'inf1_snap_ar', 'inf1_sfid_ar', 'vpeak']
            return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0], cm_ar[:, 0][0], cm_ar[:, 1][0], cm_ar[:, 2][0], this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], sfid, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], if_tree['SubhaloVmax'][if_snap == matching_snap][0], if_tree['SubhaloMass'][if_snap == matching_snap][0]*1e10/h, inf1_snap, if_tree['SubfindID'][if_snap == inf1_snap][0], vpeak, vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0], crossing_snap, crossing_sfid
            

            
        
    if i2 ==1 and i1 == 0: # This would be the case wherewe have shifting to the FoF, but it was never a central. It has to be in one of the two cases.
        return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0], cm_ar[:, 0][0], cm_ar[:, 1][0], cm_ar[:, 2][0], this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], sfid, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], if_tree['SubhaloVmax'][if_snap == matching_snap][0], if_tree['SubhaloMass'][if_snap == matching_snap][0]*1e10/h, -1, -1, vpeak, vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0], crossing_snap, crossing_sfid

    

def get_merged_ids(sfid):
    '''
    Expecting a full merger tree as the input. 
    This function returns the list of IDs which merged into an FoF halo at a given snap

    -o- The sfid that is input here is the one at snapshot 99. 
    -o- Assumptions: All the 'NextProgenitorID's are the ones which merged into a given subhalos
    -o- However, NextProgenitorID is given for the merger tree, we need to get its SubFind ID
    -o- How to get the snapshot of merger? Current snapshot is the last snapshot where the subhalos are going to exist.
    -o- We are not going to consider subhalos which merged outside the FoF halo. 
    '''
    merged_fields = ['SubhaloGrNr', 'FirstProgenitorID', 'NextProgenitorID', 'SnapNum', 'SubfindID', 'SubhaloID', 'SubhaloMassInRadType']

    tree = il.sublink.loadTree(basePath, 99, sfid, fields = merged_fields)

    merged_ids = np.zeros(0)
    merged_snaps = np.zeros(0)

    snaps = tree['SnapNum']
    grnrs = tree['SubhaloGrNr']
    sfids = tree['SubfindID']
    subids = tree['SubhaloID']
    mstar = tree['SubhaloMassInRadType'][:, 4]*1e10/h
    npids = tree['NextProgenitorID']
    npids = npids[npids != -1]
    # print(np.all(np.diff(subids) == 1))

    # print(subids)
    for ix in (range(len(npids))): # This is to loop over all the Next Progenitor IDs
        npid = npids[ix]
        this_sh_ix = npid - subids[0]
        sh_grnr = grnrs[this_sh_ix]
        sh_merger_snap = snaps[this_sh_ix]
        this_snap_central_grnr = get_grnr(sh_merger_snap)

        if sh_grnr == this_snap_central_grnr:
            # continue
            sh_merger_snap = snaps[this_sh_ix] #This would be the last snapshot where the subhalo existed
            sh_sfid = sfids[this_sh_ix] #This is the SubFind ID of the subhalo
            merged_ids = np.append(merged_ids, sh_sfid)
            merged_snaps = np.append(merged_snaps, sh_merger_snap)

    

    return merged_snaps, merged_ids 




this_fof_path = fof_path + 'fof' + str(fofno) + '_partdata/' #This is the path where the particle data for this FoF halo is stored

dm_ids = np.load(this_fof_path+'dm_ids.npy') # DM IDs
dm_pos = np.load(this_fof_path+'dm_pos.npy') # Position w.r.t. the GroupPos (MBP of the FoF halo)
dm_vel = np.load(this_fof_path+'dm_vel.npy') # Velocities w.r.t. the GroupVel (CM? of the FoF halo)


def get_positions(mbpid, mbpidp):
    '''
    This function is for parallelizing the process of finding the positions of the subhalos.
    The returned positions and velocities are already w.r.t the center of the FoF halo
    '''
    pos = [None]
    vel = [None]
    pos2 = [None]
    vel2 = [None]

    index = np.where(np.isin(dm_ids, mbpid))[0]
    # print(index)
    if len(index) == 1: 
        pos = dm_pos[index][0]
        vel = dm_vel[index][0]
    
    index2 = np.where(np.isin(dm_ids, mbpidp))[0]
    if len(index2) == 1: 
        pos2 = dm_pos[index2][0]
        vel2 = dm_vel[index2][0]     

    # In the case of having a position for MBP ID of the merger snapshot and the previous snapshot, 
    # the position would be the average of both positions, else, it is only one of these positions. 

    posavg = []
    velavg = []
    if len(pos) == 3 and len(pos2) == 3:
        posavg = np.array(pos + pos2)/2.
        velavg = np.array(vel + vel2)/2.
    elif len(pos) ==3 and len(pos2) == 1:
        posavg = np.array(pos)
        velavg = np.array(vel)
    elif len(pos2) == 3 and len(pos) == 1:
        posavg = np.array(pos2)
        velavg = np.array(vel2)
    elif len(pos2) == 1 and len(pos) == 1:
        return None

    if len(posavg) == 3:
        return posavg, velavg
    else:
        return None



def get_t2subhalo_data(merger_sfid, merger_snap, merged_into):
    '''
    This function is to obtain the subhalo data for Type-2 subhalos

    Args:
    merger_sfid: The SubFind ID of the subhalo at the merger snapshot
    merger_snap: The merger snapshot
    merged_into: The subhalo into which the merger happened

    Returns:
    Properties of the subhalos at z = 0
    '''
    # try:
    this_subh = il.groupcat.loadSingle(basePath, int(merger_snap), subhaloID = int(merger_sfid)) # Loading the subhalos at the merger snapshot
    # except OSError:
    #     logging.info('Subhalo not found at snapshot %s with SubFind ID %s', merger_snap, merger_sfid)
    #     print('Subhalo not found at snapshot %s with SubFind ID %s', merger_snap, merger_sfid)
    #     return None
    mbpid = this_subh['SubhaloIDMostbound'] # This is the most bound particle ID at this snapshot

    
    # Let us now get the vmax_if by looking at the merger tree
    if_tree = il.sublink.loadTree(basePath, int(merger_snap), int(merger_sfid),
            fields = ['SubfindID', 'SnapNum', 'SubhaloMass', 'SubhaloGrNr', 'GroupFirstSub', 'SubhaloVmax', 'SubhaloIDMostbound', 'SubhaloPos'], 
            onlyMPB = True) #Progenitor tree
    if_snap = if_tree['SnapNum'] # This has nothing to do with infall. This is an array of the snapshots where the subhalo exists
    snap_len = len(if_snap)
    if_grnr = if_tree['SubhaloGrNr']

    #FIXME: Crossing time calculations are not working. Probably because some of the central snapshots are missing. Should look for common indices of both subhalo and the central.
    # ixs_for_central = np.where(np.isin(central_snaps, if_snap))[0] # There are the indices for the central subhalo where the subhalo exists
    # central_pos = np.column_stack((central_x[ixs_for_central], 
    #                                  central_y[ixs_for_central], 
    #                                  central_z[ixs_for_central]))
    # # print(if_tree['SubhaloPos']/h/(1 + all_redshifts[if_snap]), central_pos, flush = True)
    # # print(len(if_tree['SubhaloPos']), len(if_snap), len(ixs_for_central), flush = True)
    # subh_x = if_tree['SubhaloPos'][:, 0]/h/(1 + all_redshifts[if_snap]) 
    # subh_x = subh_x - central_x[ixs_for_central] # This is already w.rt. the central subhalo
    # subh_y = if_tree['SubhaloPos'][:, 1]/h/(1 + all_redshifts[if_snap]) - central_y[ixs_for_central]
    # subh_z = if_tree['SubhaloPos'][:, 2]/h/(1 + all_redshifts[if_snap]) - central_z[ixs_for_central]

    # # if_pos = if_tree['SubhaloPos']/h/(1 + all_redshifts[if_snap]) - central_pos # This is the position of the subhalo w.r.t. the central subhalo
    # if_dist = np.sqrt(subh_x**2 + subh_y**2 + subh_z**2) # This is the distance of the subhalo from the central subhalo. We will be comparing this with the virial radius of the FOF halo later.

    mbpidp = if_tree['SubhaloIDMostbound'][if_snap == int(merger_snap - 1)] # This will be the most bound ID of the subhalo one snapshot previous to the merger snapshot

    vpeak = np.max(if_tree['SubhaloVmax']) #This is the peak value of Vmax for the subhalo across all snapshots

    i1 = 0
    i2 = 0
    inf1_snap = -1
    inf1_sfid = -1
    matching_snap = -1

    i3 = 0 # This will be the index where the subhalo enters the virial radius of the FoF halo
    crossing_snap = -1
    crossing_sfid = -1


    for ix in range(len(if_snap)):
        '''
        This loop is to go through all the snaps in order to obtain the snap where infall happened
        '''
        snap_ix = if_snap[snap_len - ix - 1] #Go in an ascending order of snapshots for this subhalo
        
        # print(if_snap, if_grnr, central_grnr)
        
        if (i1 == 0) and (if_tree['SubfindID'][ix] == if_tree['GroupFirstSub'][ix]):
            inf1_snap = if_snap[ix] #This would be the last time when it made transition from central to a satellite
            i1 = 1
            # print(snap, subid, inf1_snap)
        # if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
        if if_grnr[snap_len - ix - 1].size * central_grnr[central_snaps == snap_ix].size > 0: #What is this for? Assuming this is a check if subhalo existed at this snapshot
            if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
                matching_snap = snap_ix #This would be the first time when this entered FoF halo 0
                i2 = 1
        # if (i3 == 0) and (if_dist[snap_len - ix - 1] < Rvir[central_snaps == snap_ix]):
        #     crossing_snap = snap_ix
        #     crossing_sfid = if_tree['SubfindID'][snap_len - ix - 1]
        #     i3 = 1
        if i1*i2 == 1: # This is when the subhalo transitions into the FoF and also becomes a satellite at some point.
            result = get_positions(mbpid, mbpidp)
            if result is not None:
                pos_ar, vel_ar = result

                return pos_ar[0], pos_ar[1], pos_ar[2], this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], merger_sfid, merger_snap, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], if_tree['SubhaloVmax'][if_snap == matching_snap][0], if_tree['SubhaloMass'][if_snap == matching_snap][0]*1e10/h, inf1_snap, if_tree['SubfindID'][if_snap == inf1_snap][0], vpeak, vel_ar[0], vel_ar[1], vel_ar[2], merged_into, crossing_snap, crossing_sfid
            

            
        
    if i2 ==1 and i1 == 0: # This would be the case wherewe have shifting to the FoF, but it was never a central. It has to be in one of the two cases.
        result = get_positions(mbpid, mbpidp)
        if result is not None:
            pos_ar, vel_ar = result
            return pos_ar[0], pos_ar[1], pos_ar[2],  this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], merger_sfid, merger_snap, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], if_tree['SubhaloVmax'][if_snap == matching_snap][0], if_tree['SubhaloMass'][if_snap == matching_snap][0]*1e10/h, -1, -1, vpeak, vel_ar[0], vel_ar[1], vel_ar[2], merged_into, crossing_snap, crossing_sfid
    



    
#
# The following is for Type-1 subhalos  
# Uncomment all the following lines. Commented them for now since these have already been run before
#  =============================================================

# results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_t1subhalo_data)(ix) for ix in tqdm(range(central_id_at99 + 1, central_id_at99 + this_fof_nsubs)))
# results = [value for value in results if value is not None] #Getting rid of all the None entries

# #Let us now save all of these files in a single .csv file
# column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'cmx_ar', 'cmy_ar', 'cmz_ar', 'vmax_ar', 'mass_ar', 'len_ar', 'sfid_ar', 
#                 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'mdm_if_ar', 'inf1_snap_ar', 'inf1_sfid_ar', 'vpeak', 'velx_ar', 'vely_ar', 'velz_ar', 'crossing_snap', 'crossing_sfid']
# df = pd.DataFrame(columns=column_names)
# for ix in range(len(results)):
#     print(df)
#     print(results[ix])
#     df.loc[len(df)] = results[ix]


# df['fof'] = fofno

# #Let us now save the file
# df.to_csv(filepath + 'fofno_' + str(fofno) + '.csv', index = False)

#  =============================================================




# The following is Type-2 subhalos
#  =============================================================
this_fof_nsubs = 1 # Currently only doing only for the central subhalo     

subhalos_to_consider = np.arange(central_id_at99, central_id_at99 + this_fof_nsubs) #This is the list of subhalos into which mergers will be considered

results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_merged_ids)(ix) for ix in tqdm(subhalos_to_consider)) #FIXME: Change this to central_id_at99 (remove  +1)


merged_snaps = np.zeros(0)
merged_ids = np.zeros(0)
merged_into = np.zeros(0) # This is the subhalo into which the merger happened
for (ix, value) in enumerate(results):
    merged_snaps = np.append(merged_snaps, value[0])
    merged_ids = np.append(merged_ids, value[1])
    merged_into = np.append(merged_into, np.ones(len(value[0])) * subhalos_to_consider[ix]) 

print(merged_snaps, merged_ids)
# After getting a list of merged IDs, we will have to look for the infall of these subhalos and then get required data for these subhalos


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_t2subhalo_data)(int(merged_ids[ix]), int(merged_snaps[ix]), int(merged_into[ix])) for ix in tqdm(range(len(merged_ids))))

# results = [get_t2subhalo_data(int(merged_ids[ix]), int(merged_snaps[ix]), int(merged_into[ix])) for ix in range(len(merged_ids))]


results = [value for value in results if value is not None] # This is to remove all the entries that are None


#Let us now save all of these files in a single .csv file
column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'vmax_merger_ar_merger', 'mass_merger_ar', 'len_merger_ar', 
                'sfid_merger_ar', 'snap_merger_ar', 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'mdm_if_ar', 'inf1_snap_ar', 'inf1_sfid_ar', 'vpeak', 'velx_ar', 'vely_ar', 'velz_ar', 'merged_into', 'crossing_snap', 'crossing_sfid']

df = pd.DataFrame(columns=column_names)

# Print results completely
# print(results, flush = True)

for ix in range(len(results)):
    print(f"Shape of results[ix]: {len(results[ix])}")
    print(f"Type of results[ix]: {type(results[ix])}")
    # print(f"Contents of results[ix]: {results[ix]}")
    df.loc[len(df)] = results[ix]

df['fof'] = fofno

#Let us now save the file
df.to_csv(filepath + 'fofno_' + str(fofno) + '_t2.csv', index = False)




#  =============================================================