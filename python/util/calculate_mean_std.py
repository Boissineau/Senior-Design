import numpy as np
import math
import cv2 as cv

from tqdm import tqdm

class Calc_Mean_Std:
    @staticmethod
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