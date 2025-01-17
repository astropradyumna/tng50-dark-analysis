# This is to get the subhalos both Type-1 and Type-2 from the TNG50 hydro simulation.

import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed #This is to parallelize the code
import sys
from hydro_path_for_files import * #This is to get the paths for the files
sys.path.append(os.path.abspath('/bigdata/saleslab/psadh003/tng50/dwarf_formation'))
from tng_subhalo_and_halo import TNG_Subhalo
import logging
import gc

h = 0.6774
mdm = 4.5e5

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



def get_t1subhalo_data(sfid):  
    '''
    This is for Type-1 in the simulation
    '''
    this_subh = il.groupcat.loadSingle(basePath, 99, subhaloID = sfid)

    if this_subh['SubhaloFlag'] == 0:
        return None

    if_tree = il.sublink.loadTree(basePath, int(99), int(sfid),
            fields = ['SubfindID', 'SnapNum', 'SubhaloMass', 'SubhaloGrNr', 'GroupFirstSub', 'SubhaloVmax', 'SubhaloPos'], 
            onlyMPB = True) #Progenitor tree
    
    if if_tree is None:
        return None # This is to check if the subhalo is not in any tree
    
    cm = np.array(this_subh['SubhaloCM'])/h - (central_x[0], central_y[0], central_z[0])

    subh = TNG_Subhalo(sfid, 99, last_snap = 99)
    pos = subh.get_position_wrt_center(where = 99)

    vel = np.array(this_subh['SubhaloVel'] - (central_vx[0], central_vy[0], central_vz[0]))
    
    pos_ar = pos.reshape(1, -1)
    cm_ar = cm.reshape(1, -1)
    vel_ar = vel.reshape(1, -1)

    # Let us now get the vmax_if by looking at the merger tree
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
            # In this case matching_snap is the snap where it entered the FoF halo 0
            # Let us calculate the stellar mass at this snapshot
            mstar_infall = subh.get_mstar(where = int(matching_snap), how = 'total')



            # This is to calculate the Vmax at infall with different cases
            if subh.get_mdm(where = int(matching_snap), how = 'total') == 0:
                vmax_at_inf = 0
            else:
                fstar_at_infall = subh.get_mstar(where = int(matching_snap), how = 'vmax')[0]/subh.get_mdm(where = int(matching_snap), how = 'vmax')[0]
                if fstar_at_infall < 0.05:
                    vmax_at_inf = subh.get_vmax(where = int(matching_snap))[0]
                elif fstar_at_infall < 0.1:
                    vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'dm_dominated')[0]
                elif fstar_at_infall > 0.1:
                    vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'star_dominated')[0]

            if subh.get_mdm(where = int(99), how = 'total') == 0:
                vmax_at_99 = 0
            else:
                fstar_at_99 = subh.get_mstar(where = int(99), how = 'vmax')[0]/subh.get_mdm(where = int(99), how = 'vmax')[0]
                if fstar_at_99 < 0.05:
                    vmax_at_99 = subh.get_vmax(where = int(99))[0]
                elif fstar_at_99 < 0.1:
                    vmax_at_99 = subh.get_mx_values(where = int(99), typ = 'dm_dominated')[0]
                elif fstar_at_99 > 0.1:
                    vmax_at_99 = subh.get_mx_values(where = int(99), typ = 'star_dominated')[0]
               



            return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0],  vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0], sfid, vmax_at_99, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], vmax_at_inf
            

            
        
    if i2 ==1 and i1 == 0: # This would be the case wherewe have shifting to the FoF, but it was never a central. It has to be in one of the two cases.
        if subh.get_mdm(where = int(matching_snap), how = 'total') == 0:
                vmax_at_inf = 0
        else:
            fstar_at_infall = subh.get_mstar(where = int(matching_snap), how = 'vmax')[0]/subh.get_mdm(where = int(matching_snap), how = 'vmax')[0]
            if fstar_at_infall < 0.05:
                vmax_at_inf = subh.get_vmax(where = int(matching_snap))[0]
            elif fstar_at_infall < 0.1:
                vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'dm_dominated')[0]
            elif fstar_at_infall > 0.1:
                vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'star_dominated')[0]

        if subh.get_mdm(where = int(99), how = 'total') == 0:
            vmax_at_99 = 0
        else:
            fstar_at_99 = subh.get_mstar(where = int(99), how = 'vmax')[0]/subh.get_mdm(where = int(99), how = 'vmax')[0]
            if fstar_at_99 < 0.05:
                vmax_at_99 = subh.get_vmax(where = int(99))[0]
            elif fstar_at_99 < 0.1:
                vmax_at_99 = subh.get_mx_values(where = int(99), typ = 'dm_dominated')[0]
            elif fstar_at_99 > 0.1:
                vmax_at_99 = subh.get_mx_values(where = int(99), typ = 'star_dominated')[0]
            



        return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0],  vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0], sfid, vmax_at_99, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], vmax_at_inf#
# The following is for Type-1 subhalos  
# Uncomment all the following lines. Commented them for now since these have already been run before
#  =============================================================

# results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_t1subhalo_data)(ix) for ix in tqdm(range(central_id_at99 + 1, central_id_at99 + this_fof_nsubs)))
# results = [value for value in results if value is not None] #Getting rid of all the None entries

# #Let us now save all of these files in a single .csv file
# column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'velx_ar', 'vely_ar', 'velz_ar', 'sfid_ar', 'vmax_ar', 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar']
# df = pd.DataFrame(columns=column_names)
# for ix in range(len(results)):
#     # print(df)
#     # print(results[ix])
#     df.loc[len(df)] = results[ix]


# df['fof'] = fofno

# #Let us now save the file
# df.to_csv(filepath + 'hydrofofno_' + str(fofno) + '.csv', index = False)

#  =============================================================



# The following is for Type-2 subhalos

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


star_ids = np.load(this_fof_path+'star_ids.npy')
star_pos = np.load(this_fof_path+'star_pos.npy') #in kpc, wrt center of the fof halo
star_vel = np.load(this_fof_path+'star_vel.npy') #in km/s wrt 1000 central stellar particles of the FoF

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')
dm_vel = np.load(this_fof_path+'dm_vel.npy')


def get_positions_old(mbpid, mbpidp):
    '''
    This function is for parallelizing the process of finding the positions of the subhalos
    '''
    pos = [None]
    vel = [None]
    pos2 = [None]
    vel2 = [None]
    index = np.where(np.isin(star_ids, mbpid))[0]
    # print(index)
    if len(index) == 1: 
        pos = star_pos[index][0]
        vel = star_vel[index][0]
    if len(index) == 0:
        index = np.where(np.isin(dm_ids, mbpid))[0]
        if len(index) == 1: 
            pos = dm_pos[index][0]
            vel = dm_vel[index][0]

    index2 = np.where(np.isin(star_ids, mbpidp))[0]
    # print(index2)
    if len(index2) == 1: 
        pos2 = star_pos[index2][0]
        vel2 = star_vel[index2][0]
    if len(index2) == 0:
        index2 = np.where(np.isin(dm_ids, mbpidp))[0]
        if len(index2) == 1: 
            pos2 = dm_pos[index2][0]
            vel2 = dm_vel[index2][0]
    
    # print(pos, pos2)
    # posavg = pos + pos2 #This is the average position of the subhalo

    # In the case of having a position for MBP ID of the merger snapshot and the previous snapshot, 
        # the position would be the average of both positions, else, it is only one of these positions. 
        # It should either be a stellar particle or a DM particle
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
        # if len(pos_ar) == 0:
        #     pos_ar = posavg.reshape(1, -1)
        # else:
        #     return posavg
            # pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
    else:
        # popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file
        return None


def get_positions(mbpid, mbpidp):
    '''
    This function is for parallelizing the process of finding the positions of the subhalos
    '''
    pos = None
    vel = None
    pos2 = None
    vel2 = None

    index = np.where(np.isin(star_ids, mbpid))[0]
    if len(index) == 1: 
        pos = star_pos[index][0]
        vel = star_vel[index][0]
    elif len(index) == 0:
        index = np.where(np.isin(dm_ids, mbpid))[0]
        if len(index) == 1: 
            pos = dm_pos[index][0]
            vel = dm_vel[index][0]

    index2 = np.where(np.isin(star_ids, mbpidp))[0]
    if len(index2) == 1: 
        pos2 = star_pos[index2][0]
        vel2 = star_vel[index2][0]
    elif len(index2) == 0:
        index2 = np.where(np.isin(dm_ids, mbpidp))[0]
        if len(index2) == 1: 
            pos2 = dm_pos[index2][0]
            vel2 = dm_vel[index2][0]

    posavg = None
    velavg = None
    if pos is not None and pos2 is not None:
        posavg = (pos + pos2) / 2.
        velavg = (vel + vel2) / 2.
    elif pos is not None:
        posavg = pos
        velavg = vel
    elif pos2 is not None:
        posavg = pos2
        velavg = vel2

    if posavg is not None:
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
    this_subh = il.groupcat.loadSingle(basePath, int(merger_snap), subhaloID = int(merger_sfid)) # Loading the subhalos at the merger snapshot
    if this_subh['SubhaloFlag'] == 0: # Non cosmological origin subhalo
        return None
    
    
    mbpid = this_subh['SubhaloIDMostbound'] # This is the most bound particle ID at this snapshot
    # subh = TNG_Subhalo(merger_sfid, merger_snap, last_snap = merger_snap)

    
    # Let us now get the vmax_if by looking at the merger tree
    if_tree = il.sublink.loadTree(basePath, int(merger_snap), int(merger_sfid),
            fields = ['SubfindID', 'SnapNum', 'SubhaloMass', 'SubhaloGrNr', 'GroupFirstSub', 'SubhaloVmax', 'SubhaloIDMostbound', 'SubhaloPos'], 
            onlyMPB = True) #Progenitor tree
    if_snap = if_tree['SnapNum'] # This has nothing to do with infall. This is an array of the snapshots where the subhalo exists
    snap_len = len(if_snap)
    if_grnr = if_tree['SubhaloGrNr']

    
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
            if result is not None: # If we could find the subhalos's MBP

                pos_ar, vel_ar = result 
                subh = TNG_Subhalo(int(if_tree['SubfindID'][if_snap == matching_snap][0]), int(matching_snap), last_snap = merger_snap)
                if subh.get_mdm(where = int(matching_snap), how = 'total').size == 0 or subh.get_mstar(where = int(matching_snap), how = 'vmax').size == 0:
                    print(matching_snap, 'SFID at infall:', if_tree['SubfindID'][if_snap == matching_snap][0], 'Merged at snap', merger_snap, 'Merger SFID:', merger_sfid)
                    return pos_ar[0], pos_ar[1], pos_ar[2], vel_ar[0], vel_ar[1], vel_ar[2], merger_sfid, merger_snap, merged_into, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], vmax_at_inf, vpeak
                if subh.get_mdm(where = int(matching_snap), how = 'total') == 0:
                    vmax_at_inf = 0
                else:
                    fstar_at_infall = subh.get_mstar(where = int(matching_snap), how = 'vmax')[0]/subh.get_mdm(where = int(matching_snap), how = 'vmax')[0]
                    if fstar_at_infall < 0.05:
                        vmax_at_inf = subh.get_vmax(where = int(matching_snap))[0]
                    elif (fstar_at_infall < 0.1) & (fstar_at_infall > 0.05):
                        vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'dm_dominated')[0]
                    elif fstar_at_infall > 0.1:
                        vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'star_dominated')[0]
                    else: # probably the cases with nan
                        vmax_at_inf = -1

                return pos_ar[0], pos_ar[1], pos_ar[2], vel_ar[0], vel_ar[1], vel_ar[2], merger_sfid, merger_snap, merged_into, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], vmax_at_inf, vpeak
            

            
        
    if i2 ==1 and i1 == 0: # This would be the case wherewe have shifting to the FoF, but it was never a central. It has to be in one of the two cases.
        result = get_positions(mbpid, mbpidp)
        if result is not None:
            pos_ar, vel_ar = result
            subh = TNG_Subhalo(int(if_tree['SubfindID'][if_snap == matching_snap][0]), int(matching_snap), last_snap = merger_snap)
            if subh.get_mdm(where = int(matching_snap), how = 'total').size == 0 or subh.get_mstar(where = int(matching_snap), how = 'vmax').size == 0:
                    print(matching_snap, 'SFID at infall:', if_tree['SubfindID'][if_snap == matching_snap][0], 'Merged at snap', merger_snap, 'Merger SFID:', merger_sfid)
                    return None

            if subh.get_mdm(where = int(matching_snap), how = 'total') == 0:
                vmax_at_inf = 0
            else:
                fstar_at_infall = subh.get_mstar(where = int(matching_snap), how = 'vmax')[0]/subh.get_mdm(where = int(matching_snap), how = 'vmax')[0]
                if fstar_at_infall < 0.05:
                    vmax_at_inf = subh.get_vmax(where = int(matching_snap))[0]
                elif (fstar_at_infall < 0.1) & (fstar_at_infall > 0.05):
                    vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'dm_dominated')[0]
                elif fstar_at_infall > 0.1:
                    vmax_at_inf = subh.get_mx_values(where = int(matching_snap), typ = 'star_dominated')[0]
                else: # probably the cases with nan
                    vmax_at_inf = -1

            return pos_ar[0], pos_ar[1], pos_ar[2], vel_ar[0], vel_ar[1], vel_ar[2], merger_sfid, merger_snap, merged_into, matching_snap, if_tree['SubfindID'][if_snap == matching_snap][0], vmax_at_inf, vpeak
    

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

# print(merged_snaps, merged_ids)
# After getting a list of merged IDs, we will have to look for the infall of these subhalos and then get required data for these subhalos


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_t2subhalo_data)(int(merged_ids[ix]), int(merged_snaps[ix]), int(merged_into[ix])) for ix in tqdm(range(len(merged_ids))))


del star_ids, star_pos, star_vel, dm_ids, dm_pos, dm_vel # This is to free up some memory

gc.collect()

# results = [get_t2subhalo_data(int(merged_ids[ix]), int(merged_snaps[ix]), int(merged_into[ix])) for ix in range(len(merged_ids))]


results = [value for value in results if value is not None] # This is to remove all the entries that are None


#Let us now save all of these files in a single .csv file
column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'velx_ar', 'vely_ar', 'velz_ar', 'sfid_merger_ar', 'snap_merger_ar', 'sfid_into_ar', 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'vpeak']

df = pd.DataFrame(columns=column_names)

# Print results completely
# print(results, flush = True)

for ix in range(len(results)):
    # print(f"Shape of results[ix]: {len(results[ix])}")
    # print(f"Type of results[ix]: {type(results[ix])}")
    # print(f"Contents of results[ix]: {results[ix]}")
    df.loc[len(df)] = results[ix]

df['fof'] = fofno

#Let us now save the file
df.to_csv(filepath + 'hydrofofno_' + str(fofno) + '_t2.csv', index = False)



