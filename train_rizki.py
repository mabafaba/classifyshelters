
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

# data
resize_image_height_to = 128
resize_image_width_to  = 128
smooth                 = 1.0
test_data_fraction     = 0.15

# computation 
number_of_epochs = 40
batch_size       = 80





#################################################
# MODEL
#################################################

# LOSS FUNCTION: dice coeff not used here; cross entropy instead (see below)

# # dice coefficient
# def dice_coef(y_true, y_pred):
#     # truth as vector:
#     y_true_f = K.flatten(y_true)
#     # prediction as vector:
#     y_pred_f = K.flatten(y_pred)
#     # count predicted "1"s that are also true "1"s = count true positives
#     intersection = K.sum(y_true_f * y_pred_f)
#     # 2 * count true positives / ( count true "1"s + count predicted "1"s
#     # returns 0 for all wrong
#     # returns 0 for prediction all 0
#     # returns 1 for all correct
#     # how 'good' a value in between depends is on total % of true "1"s.
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# # loss is negative dice coefficient
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


# NETWORK

    #      - https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/      
    #      - https://blog.deepsense.ai/wp-content/uploads/2017/04/architecture_details.png
    #        Note: they use 20 input layers.
def get_unet():
    inputs = Input((resize_image_height_to, resize_image_width_to, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization(momentum=0.01)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn1)
    bn1 = BatchNormalization(momentum=0.01)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    bn2 = BatchNormalization(momentum=0.01)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization(momentum=0.01)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization(momentum=0.01)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    bn3 = BatchNormalization(momentum=0.01)(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization(momentum=0.01)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization(momentum=0.01)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    bn4 = BatchNormalization(momentum=0.01)(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization(momentum=0.01)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization(momentum=0.01)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    bn5 = BatchNormalization(momentum=0.01)(pool4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool4)
    bn5 = BatchNormalization(momentum=0.01)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    bn6 = BatchNormalization(momentum=0.01)(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization(momentum=0.01)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization(momentum=0.01)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    bn7 = BatchNormalization(momentum=0.01)(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    bn7 = BatchNormalization(momentum=0.01)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    bn7 = BatchNormalization(momentum=0.01)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    bn8 = BatchNormalization(momentum=0.01)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    bn8 = BatchNormalization(momentum=0.01)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    bn8 = BatchNormalization(momentum=0.01)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    bn9 = BatchNormalization(momentum=0.01)(up9)
    conv9 = Conv2D(62, (3, 3), activation='relu', padding='same')(up9)
    bn9 = BatchNormalization(momentum=0.01)(conv9)
    conv9 = Conv2D(62, (3, 3), activation='relu', padding='same')(conv9)
    bn9 = BatchNormalization(momentum=0.01)(conv9)
    conv9 = Conv2D(62, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy']) #modified by rizki.

    return model





def preprocess(imgs):
    # resize images
    imgs_p = np.ndarray((imgs.shape[0], resize_image_height_to, resize_image_width_to), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (resize_image_width_to, resize_image_height_to), preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

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
    # scale masks to [0, 1]
    imgs_mask_train /= 255.  

    # BUILD MODEL
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
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

    # for image, image_id in zip(imgs_train, imgs_id_test):
    #     image = (image[:, :, 0] * 255.).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_trainpred.png'), image)


# What to do when this file is run:

if __name__ == '__main__':
    train_and_predict()


