from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon, Arrow
import itertools
import os 
import albumentations as albu
import cv2

def get_image(index, data, target_size, dataset_directory, normalize = True, transformer = None):

    image = cv2.imread(os.path.join(dataset_directory, data['image_path'].values[index].strip())) #dataset_directory + data['image_path'].values[index])
    labels = np.array([(data['tlx'].values[index], data['tly'].values[index]), (data['trx'].values[index], data['try'].values[index]), (data['brx'].values[index], data['bry'].values[index]), (data['blx'].values[index], data['bly'].values[index])])
    image = cv2.resize(image, target_size)

    labels[labels>1] = 1
    labels[labels<0] = 0
    labels = labels*223

    if transformer is not None:
        try:
            outs = transformer(image = image, keypoints = labels)
        except:
            import pdb;pdb.set_trace()
        image = outs['image']
        labels = np.array(outs['keypoints'])
    
    if normalize:
        image = image/255.0
    
    labels = labels.reshape((8))/223
    
    return [image, labels]
    

def generator(data, image_wh, batch_size, dataset_directory, normalize = True, transformer = None):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(len(data))
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            x_train = np.empty([0, image_wh, image_wh, 3], dtype=np.float32)
            y_train = np.empty([0, 8], dtype=np.int32)

            for i in current_batch:
                # get an image and its corresponding color for an traffic light
                [image, color] = get_image(i, data, (image_wh, image_wh), dataset_directory, normalize = normalize, transformer = transformer)
                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                # import pdb;pdb.set_trace()
                y_train = np.append(y_train, [color], axis=0)

                # [image, color] = get_image(i, data, ())
                # x_train = np.append(x_train, [image], axis=0)
                # y_train = np.append(y_train, [color], axis=0)


            # y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

            yield (x_train, y_train)

def create_transformer(transformations):
    return albu.Compose(transformations, p=1, keypoint_params=albu.KeypointParams(format='xy'))#(image=image, keypoints=points)

def vis_points(image, points, diameter=3):
    im = image.copy()
    points = points.reshape((4, 2))

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(im)
    plt.show()