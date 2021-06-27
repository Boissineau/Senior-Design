import pyflann
import scipy.io as sio
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import imageio
import argparse
import os
import time

from tqdm import tqdm
from util.synthetic_util import SyntheticUtil
from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera
from operator import itemgetter




parser = argparse.ArgumentParser()
parser.add_argument('--feature-type', required=True, type=str, default='deep', help='deep or HoG')
parser.add_argument('--algo', required=True, type=int, default='1', help='1 - Josh or 2 - Zeke')
parser.add_argument('--load-h', required=False, default=False, help='load homographies: True or False')
args = parser.parse_args()
feature_type = args.feature_type
load_h = args.load_h
algo = args.algo
assert feature_type == 'deep' or feature_type == 'HoG'

print(load_h)

if algo == 1:
    f = open("./coordinates/pos.txt", "r")
    arr = []
    for line in f.readlines():
        line = line.strip()
        arr.append(line.split(' '))

    josh = np.array(arr, dtype=np.uint16)
    f.close()
else:
    zeke = []
    path, dirs, files = next(os.walk('./coordinates/Zeke/'))
    file_count = len(files)
    for i in range(file_count):
        f = open("./coordinates/Zeke/f" + str(i + 1) + '.txt', "r")
        arr = []
        for j, line in enumerate(f.readlines()):
            line = line.strip()
            arr.append(line.split(' '))
        f.close()
        zeke.append(np.array([np.array(x, dtype=np.uint16) for x in arr]))




# database
if feature_type == 'deep':
    # data = sio.loadmat('../data/features/database_camera_feature.mat') # use this for testing authour's data
    data = sio.loadmat('../data/my_features/database_features.mat')
    database_features = data['features']
    database_cameras = data['cameras']
else:
    data = sio.loadmat('../data/features/database_camera_feature_HoG.mat')
    database_features = data['features']
    database_cameras = data['cameras']


# testing edge image from two-GAN
if feature_type == 'deep':
    # data = sio.loadmat('../data/features/testset_feature.mat') # use this for testing authour's data
    data = sio.loadmat('../data/my_features/input.mat')
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


_, _, _, num_of_images = edge_map.shape
print(num_of_images) 



def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min



def smooth_homography(homography_list, mod_value):


    h1 = (homography_list[0])
    h2 = (homography_list[1])


    a1 = h1[0][0]
    a2 = h1[0][1]
    a3 = h1[0][2]
    b1 = h1[1][0]
    b2 = h1[1][1]
    b3 = h1[1][2]
    c1 = h1[2][0]
    c2 = h1[2][1]
    c3 = h1[2][2]

    x1 = h2[0][0]
    x2 = h2[0][1]
    x3 = h2[0][2]
    y1 = h2[1][0]
    y2 = h2[1][1]
    y3 = h2[1][2]
    z1 = h2[2][0]
    z2 = h2[2][1]
    z3 = h2[2][2]

    

    difference_1 = (x1 - a1)/mod_value
    difference_2 = (x2 - a2)/mod_value
    difference_3 = (x3 - a3)/mod_value
    difference_4 = (y1 - b1)/mod_value
    difference_5 = (y2 - b2)/mod_value
    difference_6 = (y3 - b3)/mod_value
    difference_7 = (z1 - c1)/mod_value
    difference_8 = (z2 - c2)/mod_value
    difference_9 = (z3 - c3)/mod_value




    new_h = h1
    list_of_homographies = []

    for i in range(mod_value):    
        d1 = new_h[0][0] + difference_1 
        d2 = new_h[0][1] + difference_2
        d3 = new_h[0][2] + difference_3
        e1 = new_h[1][0] + difference_4
        e2 = new_h[1][1] + difference_5
        e3 = new_h[1][2] + difference_6
        f1 = new_h[2][0] + difference_7
        f2 = new_h[2][1] + difference_8
        f3 = new_h[2][2] + difference_9

        new_h = np.array([[d1, d2, d3], [e1, e2, e3], [f1, f2, f3]])
        list_of_homographies.append(new_h)

    return list_of_homographies

def smooth_coordinates(previous_coordinates, mod_value, history, query_index):


    # smooth coordinates is only used for smoothing specifically just coordinates
    if algo == 1:
        x1 = previous_coordinates[0][0]
        y1 = previous_coordinates[0][1]
        x2 = previous_coordinates[1][0]
        y2 = previous_coordinates[1][1]
    else:
        try:
            x1 = previous_coordinates[0]
            y1 = previous_coordinates[1]
            x2 = previous_coordinates[2]
            y2 = previous_coordinates[3]
        except: 
            x2 = 0 
            y2 = 0

    difference_x = (x2 - x1)/mod_value
    difference_y = (y2 - y1)/mod_value

    # check for how much the coordinates changed so that there is no random jumping 
    if (abs(difference_x) > 5 or abs(difference_y) > 5) and mod_value != 1:
        # if high_change:
        #     # check for new change 
        #     dif_x = (high_change_coordinate[0])
    
        print(f'Large change in coordinates detected. Switching to a 0 difference. Logging coordinates:({x2:.2f}, {y2:.2f}) ')
        difference_x = 0
        difference_y = 0
        high_change_coordinate = (x2, y2)
        high_change = True

    print(f'x: {x1:.2f} to {x2:.2f} - {difference_x}\ny:{y1:.2f} to {y2:.2f} - {difference_y}')
    new_coordinate = (x1, y1)

    for i in range(mod_value):
        new_x = new_coordinate[0] + difference_x
        new_y = new_coordinate[1] + difference_y
        if new_x > 1278: 
            new_x = 1278
        if new_x < 0:
            new_x = 1
        if new_y > 718:
            new_y = 718
        if new_y < 0:
            new_y = 1
        new_coordinate = (new_x, new_y)
        history.append((new_coordinate[0], new_coordinate[1]))

        template_image = imageio.imread('test.png', pilmode='RGB')
        fig, ax = plt.subplots(1, 1, figsize=(15,10))
        ax.imshow(template_image)
        ax.scatter(new_x, new_y, s=100, c='red')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        # plt.show()
        plt.savefig(f'/home/brendan/projects/sd/SCCvSD/python/results/{query_index - mod_value + 1 + i}.jpg')
        plt.close('all')
    previous_coordinates = []
    previous_coordinates.append(history[-1]) 







    return previous_coordinates, history


def calculate_coordinates(h_list, coord, mod_value, query_index, algo, idx):
    # list_of_coordinates = []
    # for i in range(len(h_list)):
    if algo == 1:
        x = int(coord[query_index - mod_value + 1 + idx][0] + (coord[query_index - mod_value + 1 + idx][2]/2))
        y = int(coord[query_index - mod_value + 1 + idx][1] + coord[query_index - mod_value + 1 + idx][3])
    else: 
        x = int(coord[0] + coord[2]/2)
        y = int(coord[1] + coord[3])
    xy = np.stack([x, y, 1])

    H_inv = np.linalg.inv(h_list)

    w = H_inv@xy
    x_w = w[0]
    y_w = w[1]
    z_w = w[2]
    x_w /= z_w + 1e-8
    y_w /= z_w + 1e-8
    x_warped = scale_number(x_w, 0, 1280, 0, 115)
    y_warped = scale_number(y_w, 0, 720, 0, 74)

    if x_warped > 1278: 
        x_warped = 1278
    if x_warped < 0:
        x_warped = 1
    if y_warped > 718:
        y_warped = 718
    if y_warped < 0:
        y_warped = 1

    # list_of_coordinates.append((x_warped, y_warped))

    return (x_warped, y_warped)
    # return list_of_coordinates



'''

TODO

[x] clean up code
    could make function out of multi player coordinate smoothing 
[x] fix no homography found thing
[]  fix issue with # of frames != mod_value
[x] pickle homographies
[]  load homographies
[]  output coordinates
[] find out how to generate my own database images 


'''

# @profile 
def func():
    history = []    
    previous_coordinates = []
    high_change_coordinate = []
    homography_list = []
    first_pass = True
    mod_value = 20 # 20 = fps/20 = 3
    for query_index in tqdm(range(num_of_images)):

        if (query_index + 1) % mod_value == 0 or first_pass: 
            # print(f'generating homography on {query_index}')
            # Step 2: retrieve a camera using deep features
            flann = pyflann.FLANN()
            result, _ = flann.nn(database_features, test_features[query_index], 1, algorithm="kdtree", trees=8, checks=64) # This is really slow

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

            h_retrieved_to_query, error = SyntheticUtil.find_transform(retrieved_dist, query_dist) # This is really slow
            if error:
                print('setting to previous homography due to loss of transformation')
                refined_h = history[-1]
                print(type(refined_h))
            else:
                refined_h = h_retrieved_to_query@retrieved_h

            np.savetxt(f'./coordinates/homographies/{query_index}.txt', refined_h)

            if algo == 1:
                x = int(josh[query_index][0] + (josh[query_index][2]/2))
                y = int(josh[query_index][1] + josh[query_index][3])
                xy = np.stack([x, y, 1])

                H_inv = np.linalg.inv(refined_h)

                w = H_inv@xy
                x_w = w[0]
                y_w = w[1]
                z_w = w[2]
                x_w /= z_w + 1e-8
                y_w /= z_w + 1e-8
                x_warped = scale_number(x_w, 0, 1280, 0, 115)
                y_warped = scale_number(y_w, 0, 720, 0, 74)
                # print(x_warped, y_warped)
                previous_coordinates.append((x_warped, y_warped))
                homography_list.append(refined_h)
                history.append(refined_h)

            else:
                homography_list.append(refined_h)
                history.append(refined_h)

                list_of_coordinates = []
                for coord in zeke[query_index]:
                    x = int(coord[0] + (coord[2]/2))
                    y = int(coord[1] + coord[3])
                    xy = np.stack([x, y, 1])

                    H_inv = np.linalg.inv(refined_h)
                    w = H_inv@xy
                    x_w = w[0]
                    y_w = w[1]
                    z_w = w[2]
                    x_w /= z_w + 1e-8
                    y_w /= z_w + 1e-8
                    # you divide by the last component (z) to get the normalized vector from homogenous coordinates to euclidian coordinates
                    x_warped = scale_number(x_w, 0, 1280, 0, 115)
                    y_warped = scale_number(y_w, 0, 720, 0, 74)
                    list_of_coordinates.append((x_warped, y_warped, coord[4]))

                previous_coordinates.append(list_of_coordinates)


            if not first_pass:
                
                # generate smoothed coordinates between generated homographies
                np.set_printoptions(suppress=True)

                # coordinate smoothing for isngle tracking
                if algo == 1:
                    previous_coordinates, history = smooth_coordinates(previous_coordinates, mod_value, history, query_index)


                else:

                    '''
                    smoothing method: take 3 homographies, smooth between 1 and 3 with 2 as a curve. 
                    Update homographies after reaching 2nd homography point so that it stays smooth

                    '''

                    d = {}
                    for l in previous_coordinates:
                        for track in l:
                            player = track[2]
                            d.setdefault(player, [])
                            d[player].append((track[0], track[1]))
                            
                    last = previous_coordinates[-1]
                    previous_coordinates = []
                    previous_coordinates.append(last)

                    coordinate_set = []
                    for item, values in d.items():

                    
                        # print(item, values)

                        try:
                            x1 = values[0][0]
                            y1 = values[0][1]
                            x2 = values[1][0]
                            y2 = values[1][1]
                        except Exception as e:
                            continue

                        difference_x = (x2 - x1)/mod_value
                        difference_y = (y2 - y1)/mod_value

                       

                        new_coordinate = (x1, y1)

                        coordinates = []
                        for i in range(mod_value):
                            new_x = new_coordinate[0] + difference_x
                            new_y = new_coordinate[1] + difference_y
                            if new_x > 1278: 
                                new_x = 1278
                            if new_x < 0:
                                new_x = 1
                            if new_y > 718:
                                new_y = 718
                            if new_y < 0:
                                new_y = 1
                            new_coordinate = (new_x, new_y)
                            coordinates.append(new_coordinate)

                        coordinate_set.append(coordinates)


                        last = previous_coordinates[-1]
                        previous_coordinates = []
                        previous_coordinates.append(last)

                    
                    for idx in range(mod_value):

                        x = []
                        y = []
                        for i in range(len(coordinate_set)):
                            x.append(coordinate_set[i][idx][0])
                            y.append(coordinate_set[i][idx][1])

                        # template_image = imageio.imread('test.png', pilmode='RGB')
                        # fig, ax = plt.subplots(1, 1, figsize=(15,10))
                        # ax.imshow(template_image)
                        # ax.scatter(x, y, s=100, c='red')
                        # ax.axes.xaxis.set_ticks([])
                        # ax.axes.yaxis.set_ticks([])
                        # # plt.show()
                        # plt.savefig(f'/home/brendan/projects/sd/SCCvSD/python/results/{query_index - mod_value + 1 + idx}.jpg')
                        # plt.close('all')




                    
                    homography_list = []
                    homography_list.append(history[-1])


            if first_pass:
                first_pass = False            
        else:
            np.savetxt(f'./coordinates/homographies/{query_index}.txt', history[-1])

func()



f = open('./coordinates/coordinates.txt', 'w')
for c in history:
    f.write(str(int(c[0])) + ' ' + str(int(c[1])) + '\n')









# # homography smoothing for single player tracking
# h_list = smooth_homography(homography_list, mod_value)

# coordinates = []                    
# for i in range(len(h_list)):
#     coordinate = calculate_coordinates(h_list[i], josh, mod_value, query_index, algo, i)
#     coordinates.append(coordinate)

# for i in range(len(coordinates)):
#     template_image = imageio.imread('test.png', pilmode='RGB')
#     fig, ax = plt.subplots(1, 1, figsize=(15,10))
#     ax.imshow(template_image)
#     ax.scatter(coordinates[i][0], coordinates[i][1], s=100, c='red')
#     ax.axes.xaxis.set_ticks([])
#     ax.axes.yaxis.set_ticks([])
#     # plt.show()
#     plt.savefig(f'/home/brendan/projects/sd/SCCvSD/python/results/{query_index - mod_value + 1 + i}.jpg')
#     plt.close('all')

# homography_list = []
# homography_list.append(history[-1])





# homography smoothing for multi player tracking
# h_list = smooth_homography(homography_list, mod_value)
# coordinate_set = []
# for idx in range(len(h_list)):
#     coordinates = []
#     for coord in zeke[query_index]:
#         frame_coordinates = calculate_coordinates(h_list[idx], coord, mod_value, query_index, algo, idx)
#         coordinates.append(frame_coordinates)
#     coordinate_set.append(coordinates)

# for idx, i in enumerate(coordinate_set):     
    
    

#     def Extract(lst, pos):
#         return list( map(itemgetter(pos), lst ))
          
#     x = Extract(i, 0)
#     y = Extract(i, 1)



#     template_image = imageio.imread('test.png', pilmode='RGB')
#     fig, ax = plt.subplots(1, 1, figsize=(15,10))
#     ax.imshow(template_image)
#     ax.scatter(x, y, s=100, c='red')
#     ax.axes.xaxis.set_ticks([])
#     ax.axes.yaxis.set_ticks([])
#     # plt.show()
#     plt.savefig(f'/home/brendan/projects/sd/SCCvSD/python/results/{query_index - mod_value + 1 + idx}.jpg')
#     plt.close('all')