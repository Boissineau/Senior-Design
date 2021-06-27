import sys
sys.path.append('../')
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


"""
Extract feature from a siamese network
input: network and edge images
output: feature and camera
"""

parser = argparse.ArgumentParser()
parser.add_argument('--edge-image-file', required=True, type=str, help='a .mat file')
parser.add_argument('--model-name', required=True, type=str, help='model name .pth')
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--cuda-id', required=True, type=int, default=0, help='CUDA ID 0, 1, 2, 3')
parser.add_argument('--save-file', required=True, type=str, help='.mat file with')

args = parser.parse_args()
edge_image_file = args.edge_image_file


batch_size = args.batch_size
model_name = args.model_name
cuda_id = args.cuda_id
save_file = args.save_file

edge_image_dir = '/home/brendan/projects/sd/pytorch-two-GAN-master/results/soccer_seg_detection_pix2pix/test_latest/images/'
img_h = 180
img_w = 320
filenames = [img for img in glob.glob(edge_image_dir + '*.png')]
num = len(filenames)  # the number of all the test images


# 1: load edge imag
try:
    data = hdf5storage.loadmat(edge_image_file)
except FileNotFoundError:
    print('Error: can not load .mat file from {}'.format(edge_image_file))

pivot_images = data['edge_map']
pivot_images = np.transpose(pivot_images, (3, 2, 0, 1))
positive_images = pivot_images
print(pivot_images.shape)
def cal_mean_std(img_h, img_w, num, root):
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    img_list = os.listdir(root)
    img_list = [os.path.join(root, k) for k in img_list]

    for i in tqdm(range(num)):
        img = cv.imread(img_list[i])
        img = cv.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

    return means[0], stdevs[0]



print('calculating mean and std')
mean, std = cal_mean_std(img_h=img_h, img_w=img_w, num=num, root=edge_image_dir)
# mean=0.0188
# std=0.128

data_transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[mean], std=[std])])


n, c, h, w = pivot_images.shape


print('Note: assume input image resolution is 180 x 320 (h x w)')
data_loader = CameraDataset(pivot_images,
                            positive_images,
                            batch_size,
                            -1,
                            data_transform,
                            is_train=False)
print('load {} batch edge images'.format(len(data_loader)))











# 2: load network
branch = BranchNetwork()
net = SiameseNetwork(branch)

if os.path.isfile(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['state_dict'])
    print('load model file from {}.'.format(model_name))
else:
    print('Error: file not found at {}'.format(model_name))
    sys.exit()

# 3: setup computation device
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(cuda_id))
    net = net.to(device)
    cudnn.benchmark = True
print('computation device: {}'.format(device))

features = []
with torch.no_grad():
    print('generating features')
    for i in tqdm(range(len(data_loader))):
        x, _ = data_loader[i]
        x = x.to(device)
        feat = net.feature_numpy(x) # N x C

        features.append(feat)
        # append to the feature list

        # if i%100 == 0:
        #     print('finished {} in {}'.format(i+1, len(data_loader)))

features = np.vstack((features))
features = np.transpose(features)
print('feature dimension {}'.format(features.shape))

sio.savemat(save_file, {'features':features})
print('save to {}'.format(save_file))
