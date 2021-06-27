import cv2 as cv
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re


file_dir1 = '/home/brendan/projects/sd/pytorch-two-GAN-master/datasets/frames/test/'
file_dir2 = '/home/brendan/projects/sd/SCCvSD/python/r1/'
file_dir3 = '/home/brendan/projects/sd/SCCvSD/python/r2/'

def sortkey_natural(s):
    return tuple(int(part) if re.match(r'[0-9]+$', part) else part
                for part in re.split(r'([0-9]+)', s))

filenames1 = [img for img in glob.glob(file_dir1 + '*.jpg')]
filenames1.sort(key=sortkey_natural)

filenames2 = [img for img in glob.glob(file_dir2 + '*.jpg')]
filenames2.sort(key=sortkey_natural)

filenames3 = [img for img in glob.glob(file_dir3 + '*.jpg')]
filenames3.sort(key=sortkey_natural)

# print(len(filenames1), len(filenames2), len(filenames3))
idx = 0
for filename1, filename2, f3 in zip(filenames1, filenames2, filenames3):
    

    im1 = cv.imread(filename1)
    im2 = cv.imread(filename2)
    im3 = cv.imread(f3)

    h, w, _ = im2.shape
    im1 = cv.resize(im1, (w, h))


    vertical = np.concatenate((im1, im2, im3), axis=1)
    cv.imwrite(f'/home/brendan/projects/sd/SCCvSD/python/stitched/{idx}.jpg', vertical)

    print(idx)


    # im1 = plt.imread(filename1)
    # im2 = plt.imread(filename2)
    # im3 = plt.imread(f3)

    # fig, ax = plt.subplots(1, 3, figsize=(10,2))
    # ax[0].imshow(im1)
    # ax[0].axes.xaxis.set_ticks([])
    # ax[0].axes.yaxis.set_ticks([])
    
    # ax[1].imshow(im2)
    # ax[1].axes.xaxis.set_ticks([])
    # ax[1].axes.yaxis.set_ticks([])

    # ax[2].imshow(im3)
    # ax[2].axes.xaxis.set_ticks([])
    # ax[2].axes.yaxis.set_ticks([])
    
    # plt.show()
    # plt.savefig(f'/home/brendan/projects/sd/SCCvSD/python/stitched/{idx}.jpg')
    # plt.close('all')





    idx = idx + 1
