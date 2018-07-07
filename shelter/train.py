import os
import numpy as np

from skimage.io import imsave
from skimage.transform import resize

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from shelter.preprocessing.data import load_train_data

# MODEL
# every design module should have a least:
# build(): returns the actual model
# preprocess(): reshape input data
from shelter.designs import unet as design


def train(data_path,
          number_of_epochs=2,
          batch_size=10,
          test_data_fraction=0.15):

    # DATA LOADING AND PREPROCESSING
    print('Loading and preprocessing train data...')

    # load input images
    # input_path = os.path.join(data_path, 'model_input')
    input_path = data_path
    imgs_train, imgs_mask_train = load_train_data(input_path)
    imgs_train = design.preprocess(imgs_train)
    imgs_mask_train = design.preprocess(imgs_mask_train)
    imgs_train = imgs_train.astype('float32')

    # normalise data
    # this should probably happen in the preprocessing function or something.. not here
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    if std != 0:
        imgs_train /= std

    # load label masks
    imgs_mask_train = imgs_mask_train.astype('float32')
    # this should probably happen in the preprocessing function or something.. not here
    # i guess ideally we would have one preprocessing function for the input and one for the ground truths
    # scale masks to [0, 1]
    imgs_mask_train /= 255.  

    # BUILD MODEL
    print('Creating and compiling model...')
    # get_unet()
    model = design.build()
    # set up saving weights at checkpoints
    ckpt_dir = os.path.join(data_path, 'internal/checkpoints')
    ckpt_file = os.path.join(ckpt_dir, 'weights.h5')
    model_checkpoint = ModelCheckpoint(ckpt_file,
                                       monitor='val_loss',
                                       save_best_only=True)

    # FIT MODEL
    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train,
              batch_size=batch_size,
              epochs=number_of_epochs,
              verbose=1,
              shuffle=True,
              validation_split=test_data_fraction,
              callbacks=[model_checkpoint])


# What to do when this file is run:
if __name__ == '__main__':
    data_path = '/media/data/180505_v1/'

    train(data_path)


