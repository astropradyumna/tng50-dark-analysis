from joblib import Parallel, delayed #This is to parallelize the code
from tqdm import tqdm
import numpy as np
import illustris_python as il
import sys
from paths_for_files import *

# basePath = '/bigdata/saleslab/psadh003/TNG50-1-Dark/output'
# fof_path = '/bigdata/saleslab/psadh003/tng50dark/fof_partdata/'
fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

h = 0.6774

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
r200 = this_fof['Group_R_Crit200']/h


this_fof_path = fof_path + fof_str + '_partdata/'
dm_pos = np.load(this_fof_path+'dm_pos.npy')
dm_dist = np.sqrt(np.sum(dm_pos**2, axis=1))
rpl = np.logspace(1, np.log10(2 * r200), 100)

def get_Ndm(ix):
    ms = rpl[ix]
    return(len(dm_dist[dm_dist < ms]))

results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_Ndm)(ix) for ix in tqdm(range(len(rpl))))


print('Ndm', results)