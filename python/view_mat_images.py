import cv2 as cv
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# data = sio.loadmat('../data/features/testset_feature2.mat')
data = sio.loadmat('../data/train_data_10k.mat')
# images = data['edge_map'] # (720, 1280, 1, 186)
images = data['pivot_images'] # (10000, 1, 180, 3202)
images2 = data['positive_images']

h, w, _, n = images.shape

# for i in range(n):
#     image = images[:,:,:,i]
#     image2 = images2[:,:,:,i]

#     cv.imshow('img', image)
#     cv.imshow('img2', image2)
#     cv.waitKey(0)

for i, j in zip(images, images2):

    fig, ax = plt.subplots(1, 2, figsize=(15,10))
    ax[0].imshow(np.transpose(i, (1, 2, 0)))
    ax[0].axes.xaxis.set_ticks([])
    ax[0].axes.yaxis.set_ticks([])
    ax[1].imshow(np.transpose(j, (1, 2, 0)))
    ax[1].axes.xaxis.set_ticks([])
    ax[1].axes.yaxis.set_ticks([])
    plt.show()
    plt.close('all')