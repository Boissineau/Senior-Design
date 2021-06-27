import sys
import numpy as np
sys.path.append('../')

import scipy.io as sio
from util.synthetic_util import SyntheticUtil

data = sio.loadmat('../../data/worldcup_sampled_cameras.mat')
pivot_cameras = data['pivot_cameras']
positive_cameras = data['positive_cameras']

n = 91173  # change this number to set training dataset
save_file = '/home/brendan/projects/sd/SCCvSD/data/train_data_91k.mat'
pivot_cameras = pivot_cameras[0:n, :]
positive_cameras = positive_cameras[0:n,:]



data = sio.loadmat('../../data/worldcup2014.mat')
# print(data.keys())
model_points = data['points']
model_line_index = data['line_segment_index']

pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                         model_points, model_line_index)

print('{} {}'.format(pivot_images.shape, positive_images.shape))


import h5py
import hdf5storage
data = {'pivot_images':pivot_images,
                                  'positive_images':positive_images,
                                   'cameras':pivot_cameras}
hdf5storage.savemat(save_file, data, format='7.3')

print('save training file to {}'.format(save_file))