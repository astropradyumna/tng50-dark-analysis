# This code is to extract the particle data within 3rvir of the FoF
# This particle data will later be use to get the position and velocity of Type-2 subhalos using the MBP


import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import math
import pdb
import sys
import os
from illustris_python.snapshot import getSnapOffsets, snapPath, getNumPart
from illustris_python.util import partTypeNum
import h5py
from paths_for_files import *

h_small = 0.6744 #0.6744

def fold_pos(x, lbox = 35000.0 / h_small):
    '''
    This is to account for the preiodic box condition
    '''
    x = np.abs(x)
    aux = x > lbox / 2.
    x[aux] = x[aux] - lbox
    return x




snpz0 = 99
g = 4.3e-6

# We will be taking input from the terminal for the FoF group number. 
fof_no = int(sys.argv[1]) #This would be the FoF number that we would be getting the files for
haloID = fof_no
fof_str = 'fof' + str(fof_no)


this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
group_pos = this_fof['GroupPos'] / h_small
group_vel = this_fof['GroupVel']
r200 = this_fof['Group_R_Crit200'] / h_small


### output will be stored into the following folder
outpath = fof_path + fof_str + '_partdata/' 

if os.path.exists(outpath + 'dm_pos.npy') and os.path.exists(outpath + 'dm_vel.npy') and os.path.exists(outpath + 'dm_ids.npy'):
    print('Data already exists for FoF group number ' + str(fof_no))
    sys.exit()

if not os.path.exists(outpath): #If the directory does not exist, then just create it!
    os.makedirs(outpath)




subset = getSnapOffsets(basePath, 99, 0, "Subhalo")

with h5py.File(snapPath(basePath, 99), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

startpoint=0
chunk_len=500000
partType = 'dm'

dm_pos_ar = np.array(([np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]))
dm_vel_ar = np.array(([np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]))
dm_id_ar = np.zeros(0)
while(startpoint < nPart[partTypeNum(partType)]):
# while(startpoint < 500000 * 10):
    len_range=np.min([chunk_len, nPart[partTypeNum(partType)]-startpoint])
    subset['offsetType'][partTypeNum(partType)] = startpoint
    subset['lenType'][partTypeNum(partType)]    = len_range
    dm = il.snapshot.loadSubset(basePath, 99, 'dm', fields=['Velocities','Coordinates', 'ParticleIDs'], subset=subset)
    startpoint=startpoint+chunk_len #updating the startpoint here
    pos = dm['Coordinates']/h_small
    dist = np.sqrt(fold_pos(pos[:,0] - group_pos[0])**2 + fold_pos(pos[:,1] - group_pos[1])**2 + fold_pos(pos[:,2] - group_pos[2])**2)
    w = np.where(dist < 3*r200)[0]
    if len(w) > 0:
        posx_wrt_fof = fold_pos(pos[w, 0] - group_pos[0])
        posy_wrt_fof = fold_pos(pos[w, 1] - group_pos[1])
        posz_wrt_fof = fold_pos(pos[w, 2] - group_pos[2])
        pos_wrt_fof = np.column_stack((posx_wrt_fof, posy_wrt_fof, posz_wrt_fof))
        dm_pos_ar = np.vstack((dm_pos_ar, pos_wrt_fof))
        vel = dm['Velocities']
        vel_wrt_fof = vel[w] - group_vel
        dm_vel_ar = np.vstack((dm_vel_ar, vel_wrt_fof))
        dm_id_ar = np.append(dm_id_ar, dm['ParticleIDs'][w])


dm_pos_ar = dm_pos_ar[2:, :]
dm_vel_ar = dm_vel_ar[2:, :]

# Let us now save these as .npy files
np.save(outpath + 'dm_pos.npy', dm_pos_ar)
np.save(outpath + 'dm_vel.npy', dm_vel_ar)
np.save(outpath + 'dm_ids.npy', dm_id_ar)


