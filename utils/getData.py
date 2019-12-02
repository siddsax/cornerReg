import tensorflow as tf
import pandas as pd
import numpy as np
import os 

import albumentations as albu
from dataHelpers import generator, create_transformer

def getData(params):

    target_size = (params.image_wh, params.image_wh)
    seed = 1

    lst = [x[0].split('/')[-1] for x in os.walk(params.dataset_directory)]

    if 'train' in lst:
        dataset_directoryTR = params.dataset_directory + '/train'
        dataset_directoryTE = params.dataset_directory + '/test'
        
        trainDF = pd.read_csv(os.path.join(dataset_directoryTR, 'labels.csv'), header='infer')
        testDF = pd.read_csv(os.path.join(dataset_directoryTE, 'labels.csv'), header='infer')

        trainDF.drop(columns=['glare', 'fld_mask', 'punch'], inplace=True)
        testDF.drop(columns=['glare', 'fld_mask', 'punch'], inplace=True)

        train_len = len(trainDF)
        filenames = list(testDF)[0]
        labels = list(trainDF)[1:]

        steps_per_epoch = len(trainDF) // params.batch_size

    else:
        df = pd.read_csv(os.path.join(params.dataset_directory, 'labels.csv'), header='infer')
        df.drop(columns=['glare'], inplace=True)

        labels = list(df)[1:]
        filenames = list(df)[0]
        train_len = len(df) // 2
        valid_len = len(df) * 3 // 4

        trainDF = df[:train_len]
        testDF = df[valid_len:]

        dataset_directoryTR = dataset_directory
        dataset_directoryTE = dataset_directory

        steps_per_epoch = len(df) // params.batch_size

    if params.albumentations:
        transformer = create_transformer([
                                        albu.VerticalFlip(p=.5), 
                                        albu.HorizontalFlip(p=0.5),
                                        # albu.Flip(p=0.5),
                                        albu.OneOf([albu.HueSaturationValue(p=0.5), albu.RGBShift(p=0.7)], p=1),
                                        albu.RandomBrightnessContrast(p=0.5)
                                        ])
    else:
        transformer = create_transformer([])

    factor = 1/255. if params.normalize else 1.0
    train_generator = generator(trainDF, params.image_wh, params.batch_size, dataset_directoryTR, normalize = params.normalize, transformer = transformer)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, rescale=factor, horizontal_flip=False, vertical_flip=False)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testDF,
        directory=dataset_directoryTE,
        x_col=filenames,
        y_col=labels,
        batch_size=1,
        seed=seed,
        shuffle=False,
        class_mode="other",
        target_size=target_size)

    image_batch, label_batch = next(train_generator)
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)

    end_step = np.ceil(1.0 * train_len / batch_size).astype(np.int32) * params.epochs

    return train_generator, test_generator, end_step
