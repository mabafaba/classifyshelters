import numpy as np

from skimage.transform import resize

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from shelter.designs.components.loss_functions import dice_coef_loss
from shelter.designs.components.loss_functions import dice_coef


# CITATION
# - U-Net: https://arxiv.org/pdf/1505.04597.pdf
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

    print('using model: unet64filters') 

    # expected input shape
    inputs = Input((resize_image_height_to, resize_image_width_to, 1)) #  1 channel, x rows, y = x columns

    # convolution
    # Conv2D(number of filters, (kernel X, kernel Y), .. )
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) # -> convolution to  features: 32     window: 3
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)  # -> convolution to  features: 32     window: 9
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                         # -> maxpool to      features: 32     image : x / ( 2 )

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # -> convolution to  features: 64     window: 18
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)  # -> convolution to  features: 64     window: 18
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                         # -> maxpool to      features: 64     image : x / (2^2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2) # -> convolution to  features: 128    window: 54 
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) # -> convolution to  features: 128    window: 162
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                         # -> maxpool to      features: 128    image : x / (2^3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3) # -> convolution to  features: 256    window: 486
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4) # -> convolution to  features: 256    window: 1458
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)                         # -> maxpool to      features: 256    window: ..

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool4) # -> convolution to  features: 512    window: ..
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) # -> convolution to  features: 512    image : x / (2^4)


    # deconvolution
    up6     = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5) # deconv to  features: 512    x / (2^4)
    concat6 = concatenate([up6, conv4], axis=3)                                   # add conv4
    conv6   = Conv2D(64, (3, 3), activation='relu', padding='same')(concat6)     # convolute
    conv6   = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)      # convolute

    up7     = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = concatenate([up7, conv3], axis=3)
    conv7   = Conv2D(64, (3, 3), activation='relu', padding='same')(concat7)
    conv7   = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8     = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8   = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8   = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # dont know why but i had commented this line.. which stopped loading existing weights (?)
    # aha: IF YOU HAVE NO WEIGHTS YET YOU NEED TO UNCOMMENT THIS LINE
    # when you run the the n>1th time copy the weights.h5 file from /output/ to /checkpoints/
    # model.load_weights("weights.h5")
    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])

    return model



