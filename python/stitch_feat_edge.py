import cv2 as cv
import scipy.io as sio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time
import sys
import torch
import re 
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import h5py
import hdf5storage
from torch.utils.data import Dataset

from PIL import Image

from deep.siamese import BranchNetwork, SiameseNetwork

edge_map = []
edge_image_dir = '/home/brendan/projects/sd/pytorch-two-GAN-master/results/soccer_seg_detection_pix2pix/test_latest/images/'
feature_file = '../data/my_features/feature_file.mat'
height = 720
width = 1280


try:
    data = hdf5storage.loadmat(feature_file)
except FileNotFoundError:
    print('Error: can not load .mat file from {}'.format(feature_file))
features = data['features']

def sortkey_natural(s):
    return tuple(int(part) if re.match(r'[0-9]+$', part) else part
                for part in re.split(r'([0-9]+)', s))

filenames = [img for img in glob.glob(edge_image_dir + '*.png')]
filenames.sort(key=sortkey_natural)
# print(filenames)

for filename in filenames:
	image = cv.imread(filename, 0)
	image = cv.resize(image, (width, height))
	image = image.reshape(image.shape + (1,))
	# cv.imshow('frame', image)
	# cv.waitKey(0)
	# print(image.shape)
	edge_map.append(image)


edge_map = np.array(edge_map).transpose(1, 2, 3, 0)
print(edge_map.shape)

save_file = "../data/my_features/input.mat"
sio.savemat(save_file, {'edge_map': edge_map, 'features':features})
