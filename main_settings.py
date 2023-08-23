import os

# TODO: provide your own folders
dataset_folder = '/cluster/scratch/denysr/dataset/'
tmp_folder = '/cluster/scratch/denysr/eval/'

g_tbd_folder = dataset_folder+'TbD/'
g_tbd3d_folder = dataset_folder+'TbD-3D/'
g_falling_folder = dataset_folder+'falling_objects/'
g_wildfmo_folder = dataset_folder+'wildfmo/'
g_youtube_folder = dataset_folder+'youtube/'

g_syn_folder = dataset_folder+'synthetic/'
g_bg_folder = dataset_folder+'vot2018.zip'

g_ext_folder = dataset_folder+'s2dnet_weights.pth'
g_raft_model = dataset_folder+'raft_models/models/raft-things.pth'
g_alpha_model = dataset_folder+"ostrack/SEcmnet_ep0040-c.pth.tar"

g_resolution_x = int(640/2)
g_resolution_y = int(480/2)