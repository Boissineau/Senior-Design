import sys
import numpy as np
sys.path.append('../')

import sys
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import time
import numpy as np
import scipy.io as sio

import argparse

import h5py
import hdf5storage
import glob
import re
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from siamese import BranchNetwork, SiameseNetwork
from camera_dataset import CameraDataset

from tqdm import tqdm
from util.synthetic_util import SyntheticUtil
from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera
from operator import indexOf, itemgetter


from util.rotation_util import RotationUtil
from util.projective_camera import ProjectiveCamera




'''

Input: PTZ camera base information
Output: randomly sampled camera parameters
:param cc_statistics:
:param fl_statistics:
:param roll_statistics:
:param pan_range:
:param tilt_range:
:param u:
:param v:
:param camera_num:
:return: N * 9 cameras
returns [width, height, fl, rot_vec[0], rot_vec[1], rot_vec[2], cc[0], cc[1], cc[2]]

'''

# figure out how to determine these camera parameters
# VXL? 
# https://github.com/lood339/SCCvSD/issues/4
data = sio.loadmat('../../data/worldcup_dataset_camera_parameter.mat')
cc_mean = data['cc_mean']
cc_std = data['cc_std']
cc_min = data['cc_min']
cc_max = data['cc_max']

fl_mean = data['fl_mean']
fl_std = data['fl_std']
fl_min = data['fl_min']
fl_max = data['fl_max']

cc_statistics = [cc_mean, cc_std, cc_min, cc_max]
fl_statistics = [fl_mean, fl_std, fl_min, fl_max]
roll_statistics = [0, 0.2, -1.0, 1.0]
pan_range = [-35.0, 35.0]
tilt_range = [-15.0, -5.0]

# volleyball parameters
pan_range = [-20.0, 20.0]
tilt_range = [-20.0, -15.0]
cc_statistics = [[10, -8 ,4], [0.5,2,1], [9, -10,1], [11,-2,5]]
fl_statistics = [2600,100,2500,2700]






'''
[ 640.          360.         2672.31852665    1.87465417    0.0282665
   -0.02099061   10.47437779   -8.89159491    3.8102537 ]
'''





# w, h, fl, x, y, z, cc[x], cc[y], cc[z]
# lower fl means less zoomed
# cc[z] is height
# cc[y] is distance forward/backward. Negative means backwards
# cc[x] left/right from origin 0,0

num_camera = 91173

# generating cameras using the camera parameters from above
pivot_cameras = SyntheticUtil.generate_ptz_cameras(cc_statistics,
                                             fl_statistics,
                                             roll_statistics,
                                             pan_range, tilt_range,
                                             1280/2.0, 720/2.0,
                                             num_camera)

"""
Sample a camera that has similar pan-tilt-zoom with (pan, tilt, fl).
The pair of the camera will be used a positive-pair in the training
:param pp: [u, v]
:param cc: camera center
:param base_roll: camera base, roll angle
:param pan:
:param tilt:
:param fl:
:param pan_std:
:param tilt_std:
:param fl_std:
:return:
"""
# returns [width, height, fl, rot_vec[0], rot_vec[1], rot_vec[2], cc[0], cc[1], cc[2]]

pan_std = np.std(np.random.uniform(-35.0, 35.0, num_camera))
tilt_std = np.std(np.random.uniform(-15.0, -5.0, num_camera))

f_std = np.std(pivot_cameras[:, 2])
x_std = np.std(pivot_cameras[:, 3])
y_std = np.std(pivot_cameras[:, 4])
z_std = np.std(pivot_cameras[:, 5])

positive_cameras = []
import random
inc = 0
for camera in pivot_cameras:
    fl = camera[2] + (random.uniform(-0.5, 0.5) * f_std)
    x = camera[3] + (random.uniform(-0.5, 0.5) * x_std)
    y = camera[4] + (random.uniform(-0.5, 0.5) * y_std)
    z = camera[5] + (random.uniform(-0.5, 0.5) * z_std)

    c = np.zeros(9)
    c[0] = 640.0
    c[1] = 360.0 
    c[2] = fl
    c[3] = x
    c[4] = y
    c[5] = z
    c[6] = camera[6]
    c[7] = camera[7]
    c[8] = camera[8]

    positive_cameras.append(c)

positive_cameras = np.array(positive_cameras)

# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 3. Generate points and line_index of the field

'''
worldcup2014.mat
points: N * 2, points in the model, for example, line intersections
line_segment_index: N * 2, line segment start/end point index
grid_points: a group of 2D points uniformly sampled inside of the playing ground.
             It is used to approximate the area of the playing ground.

''' 

def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

# data = sio.loadmat('../../data/worldcup2014.mat')
# model_points = data['points']
# model_line_index = data['line_segment_index']

# x = np.loadtxt('../../data/field_map2.mat')

# x = x[0:18]
# data = {'line_segment_index':model_line_index[0:10],
#                                   'points':x}
# hdf5storage.savemat('../../data/field_map3.mat', data, format='7.3')

points = []
segments = []

image = plt.imread('../volleyball.png')
image.resize()
fig, ax = plt.subplots()
ax.imshow(image, origin='lower')
# ax.scatter(model_points[:, 0], model_points[:, 1])
def onclick(event):
    print(points)
    points.append((int(event.xdata), int(event.ydata)))
    plt.scatter(event.xdata, event.ydata, c='r')
    fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', onclick)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
# plt.show()
points = [(2,3),(161,  3),(161,  3),(246,  2),(246,  2),(333,  3),(333,  3),(491,  3),(  2,245),(161,245),(161,245),(247,244),(247,244),(333,245),(333,245),(492,246),(  3,  2),(  2,244),(492,  3),(492,245),(161,  3),(161,245),(248,  2),(249,243),(334,  3),(334,244)]
tmp = []
for i in points:
    x = int(scale_number(i[0], 0, 18, 0, 495))
    y = int(scale_number(i[1], 0, 9, 0, 248))
    tmp.append((x, y))
points = tmp

for i in range(len(points)):
    if i % 2 == 0:
        continue
    segments.append((i, i+1))


points = np.array(points)
segments = np.array(segments)
segments = segments[:]-1


data = hdf5storage.loadmat('../../data/worldcup2014.mat')
# data = hdf5storage.loadmat('../../data/field_map.mat')
model_points = data['points']
model_line_index = data['line_segment_index']


# pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                         # model_points, model_line_index)
pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                         points, segments)

# plt.scatter(model_points[:, 0], model_points[:, 1])
# plt.show()

for piv, pos in zip(pivot_cameras, positive_cameras):
    im1 = SyntheticUtil.camera_to_edge_image(piv, model_points, model_line_index, 720, 1280)
    im2 = SyntheticUtil.camera_to_edge_image(pos, model_points, model_line_index, 720, 1280)
    cv.imshow("pivot", im1)
    cv.imshow("positive", im2)
    cv.waitKey(5000)









print('{} {}'.format(pivot_images.shape, positive_images.shape))



data = {'pivot_images':pivot_images,
                                  'positive_images':positive_images,
                                   'cameras':pivot_cameras}
# hdf5storage.savemat(save_file, data, format='7.3')

print('save training file to {}'.format(save_file))


