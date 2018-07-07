import numpy as np

from skimage.transform import resize

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from shelter.designs.components.loss_functions import dice_coef_loss


# CITATION
# - https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
# - https://blog.deepsense.ai/wp-content/uploads/2017/04/architecture_details.png
#   Note: they use 20 input layers.


# MODEL PARAMETERS
smooth = 1.0
resize_image_height_to = 128
resize_image_width_to = 128


# PREPARING DATA
def preprocess(imgs):
    # resize images
    imgs_p = np.ndarray((imgs.shape[0],
                         resize_image_height_to,
                         resize_image_width_to),
                        dtype=np.uint8)

    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i],
                           (resize_image_width_to, resize_image_height_to),
                           preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]

    return imgs_p


# MODEL
def build():

    print('using model: flatunet') 

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

    # model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy']) #modified by rizki.
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['binary_accuracy'])

    return model
