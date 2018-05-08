
#################################################
# DEPENDENCIES
#################################################

# libraries
from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import BatchNormalization 

# package parameters
from data import load_train_data, load_test_data
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


#################################################
# PARAMETERS
#################################################

number_of_epochs = 40
batch_size       = 80
test_data_fraction     = 0.15




#################################################
# MODEL
#################################################
	# every model file should have at least:
		# preprocess() <- a function for data preprocessing
		# get_model() <- a function to return the model
		# now we load the new rizki version that removes filters and adds batch normalisation:

import 0002_unet_fewer_filters


def train_and_predict():

    #################
    # TRAIN
    #################

    # DATA LOADING AND PREPROCESSING
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # load input images
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    imgs_train = imgs_train.astype('float32')

    # normalise data
    # this should probably happen in the preprocessing function or something.. not here
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    if std!=0:
        imgs_train /= std

    # load label masks
    imgs_mask_train = imgs_mask_train.astype('float32')
    # this should probably happen in the preprocessing function or something.. not here
    # i guess ideally we would have one preprocessing function for the input and one for the ground truths
    # scale masks to [0, 1]
    imgs_mask_train /= 255.  

    # BUILD MODEL
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # get_unet()
    model = get_model()
    # set up saving weights at checkpoints
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    # FIT MODEL
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=number_of_epochs, verbose=1, shuffle=True,
              validation_split=test_data_fraction,
              callbacks=[model_checkpoint])


    #################
    # TEST
    #################

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


    # i forgot what this does exactly but i had commented it out at some point.
    # looks like it's scaling a matrix back up to 0-255 and saves it as png.
    # i think it's predicting the data it was trained on:

    # for image, image_id in zip(imgs_train, imgs_id_test):
    #     image = (image[:, :, 0] * 255.).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_trainpred.png'), image)







# What to do when this file is run:

if __name__ == '__main__':
    train_and_predict()


