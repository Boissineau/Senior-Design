import pyflann
import scipy.io as sio
import numpy as np
import cv2 as cv
import time
import imageio
import matplotlib.pyplot as plt
from util.synthetic_util import SyntheticUtil
from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature-type', required=True, type=str, default='deep', help='deep or HoG')
parser.add_argument('--query-index', required=True, type=int, default=0, help='[0, 186)')

args = parser.parse_args()
feature_type = args.feature_type
assert feature_type == 'deep' or feature_type == 'HoG'
query_index = args.query_index
# assert 0 <= query_index < 186

"""
Estimate an homogrpahy using edge images 
"""

# Step 1: load data
# database
if feature_type == 'deep':
    data = sio.loadmat('../data/features/database_camera_feature.mat')
    database_features = data['features']
    database_cameras = data['cameras']
    # data = sio.loadmat('../data/features/feature_camera_90k.mat') # same thing
    # database_features = data['features']
    # database_cameras = data['cameras']
else:
    data = sio.loadmat('../data/features/database_camera_feature_HoG.mat')
    database_features = data['features']
    database_cameras = data['cameras']

# testing edge image from two-GAN
if feature_type == 'deep':
    data = sio.loadmat('../data/features/testset_feature.mat')
    edge_map = data['edge_map']
    test_features = data['features']
    test_features = np.transpose(test_features)
else:
    data = sio.loadmat('../data/features/testset_feature_HoG2.mat')
    edge_map = data['edge_map']
    test_features = data['features']

# World Cup soccer template
data = sio.loadmat('../data/worldcup2014.mat')
model_points = data['points']
model_line_index = data['line_segment_index']

template_h = 74  # yard, soccer template
template_w = 115


# ground truth homography
data = sio.loadmat('../data/UoT_soccer/test.mat')
annotation = data['annotation']
# gt_h = annotation[0][query_index][1]  # ground truth


state_time = time.time()
# Step 2: retrieve a camera using deep features
flann = pyflann.FLANN()
result, _ = flann.nn(database_features, test_features[query_index], 1, algorithm="kdtree", trees=8, checks=64)
retrieved_index = result[0]


"""
Retrieval camera: get the nearest-neighbor camera from database
"""
retrieved_camera_data = database_cameras[retrieved_index]

u, v, fl = retrieved_camera_data[0:3]
rod_rot = retrieved_camera_data[3:6]
cc = retrieved_camera_data[6:9]

retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

retrieved_h = IouUtil.template_to_image_homography_uot(retrieved_camera, template_h, template_w)

# iou_1 = IouUtil.iou_on_template_uot(gt_h, retrieved_h)
# print('retrieved homogrpahy IoU {:.3f}'.format(iou_1))

retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, model_points, model_line_index,
                                               im_h=720, im_w=1280, line_width=4)

query_image = edge_map[:,:,:,query_index]
# cv.imshow('Edge image of query image', query_image)
# cv.imshow('Edge image of retrieved camera', retrieved_image)
# cv.waitKey(0)

"""
Refine camera: refine camera pose using Lucas-Kanade algorithm 
"""
dist_threshold = 50
query_dist = SyntheticUtil.distance_transform(query_image)
retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)

query_dist[query_dist > dist_threshold] = dist_threshold
retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

# cv.imshow('Distance image of query image', query_dist.astype(np.uint8))
# cv.imshow('Distance image of retrieved camera', retrieved_dist.astype(np.uint8))
# cv.waitKey(0)

h_retrieved_to_query = SyntheticUtil.find_transform(retrieved_dist, query_dist)

refined_h = h_retrieved_to_query@retrieved_h

def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

# x = scale_number(519, 0, 115, 0, 720)
# y = scale_number(284, 0, 74, 0, 1280)
x = 360 # or 320
y = 407 # or 180
xy = np.stack([x, y, 1])
print(xy)

H_inv = np.linalg.inv(refined_h)

w = H_inv@xy
x_w = w[0]
y_w = w[1]
z_w = w[2]
x_w /= z_w + 1e-8
y_w /= z_w + 1e-8
x_warped = scale_number(x_w, 0, 1280, 0, 115)
y_warped = scale_number(y_w, 0, 720, 0, 74)
print(x_warped, y_warped)


template_image = imageio.imread('test.png', pilmode='RGB')
fig, ax = plt.subplots(1, 3, figsize=(15,10))
ax[2].imshow(template_image)
ax[2].scatter(x_warped, y_warped)
ax[2].axes.xaxis.set_ticks([])
ax[2].axes.yaxis.set_ticks([])
ax[0].imshow(query_image)
ax[0].scatter(x, y)
ax[0].axes.xaxis.set_ticks([])
ax[0].axes.yaxis.set_ticks([])
ax[1].imshow(retrieved_image)
ax[1].scatter(x, y)
ax[1].axes.xaxis.set_ticks([])
ax[1].axes.yaxis.set_ticks([])
plt.show()



# iou_2 = IouUtil.iou_on_template_uot(gt_h, refined_h)
# print('refined homogrpahy IoU {:.3f}'.format(iou_2))
# print('takes time {:.3f} seconds'.format(time.time()-state_time))


