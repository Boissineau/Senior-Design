import numpy as np
import glob 
import re
import cv2 
import hdf5storage



data_dir = './data/calib/'

def sortkey_natural(s):
    return tuple(int(part) if re.match(r'[0-9]+$', part) else part
                for part in re.split(r'([0-9]+)', s))

filenames = [file for file in glob.glob(data_dir + '*.npy')]
filenames.sort(key=sortkey_natural)
np.set_printoptions(suppress=True)


a_list = [] 
r_list = []
t_list = []


for file in filenames:
    a = np.load(file, allow_pickle=True)
    recdict = a.tolist()
    a_list.append(recdict['A'])
    r_list.append(recdict['R'])
    t_list.append(recdict['T'])
# print(a_list[0])
# print()
# print(r_list[0])
# print()
# print(t_list[0])

# rt = np.array(r_list[0]).transpose()
# cp = np.linalg.inv(rt) * np.array(t_list[0])
# print(cp)

cworld = np.array(r_list[0]) * np.array(t_list[0])
print(cworld)


fl = []
cc = []
for i, j in zip(a_list, t_list):
    fl.append(i[0][0])
    cc.append(j)

fl_std = np.std(fl)
fl_min = np.min(fl)
fl_mean = np.mean(fl)
fl_max = np.max(fl)

print()
ccl = np.array(cc)
ccx = ccl[:, 0]
ccy = ccl[:, 1]
ccz = ccl[:, 2]


cc_std = np.array((np.std(ccx), np.std(ccy), np.std(ccz)))
cc_mean = np.array((np.mean(ccx), np.mean(ccy), np.mean(ccz)))
cc_min = np.array((np.min(ccx), np.min(ccy), np.min(ccz)))
cc_max = np.array((np.max(ccx), np.max(ccy), np.max(ccz)))


print(fl_max)
print()
print(fl_mean)
print()
print(fl_min)
print()
print(fl_std)


# cc_std = np.array([2.2319, 9.3826, 2.9488])
# cc_min = np.array([45.057, -66.070, 10.139])
# cc_mean = np.array([52.366, -45.157, 16.822])
# cc_max = np.array([60.846, -16.742, 23.011])



savefile = '../../data/custom_camera_parameters.mat'
hdf5storage.savemat(savefile, {'cc_max' : cc_max, 'cc_mean' : cc_mean, 'cc_min' : cc_min, 'cc_std' : cc_std, 'fl_max' : fl_max, 'fl_mean' : fl_mean, 'fl_min' : fl_min, 'fl_std' :fl_std})


