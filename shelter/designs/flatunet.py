import numpy as np

from skimage.transform import resize

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from shelter.designs.components.loss_functions import dice_coef_loss
from shelter.designs.components.loss_functions import dice_coef

# CITATION
# - https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
# - https://deepsense.ai/wp-content/uploads/2017/04/architecture_overview.png
# - https://deepsense.ai/wp-content/uploads/2017/04/architecture_details.png
#   Note: they use 20 input layers.
#   Note: We begin at 128 * 128, so 1 less U-layer from reference.

# resize input matrix
resize_image_height_to = 128
resize_image_width_to = 128


# MODEL
def build():

    print('using model: mod-unet (https://deepsense.ai/wp-content/uploads/2017/04/architecture_details.png)') #based on

    #h * w = 128 * 128
    inputs = Input((resize_image_height_to, resize_image_width_to, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

    #h * w = 128 * 128
    bn2 = BatchNormalization(momentum=0.01)(conv1)
    bn_conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization(momentum=0.01)(bn_conv2_1)
    bn_conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(bn_conv2_2)

    #h * w = 64 * 64
    bn3 = BatchNormalization(momentum=0.01)(pool2)
    bn_conv3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization(momentum=0.01)(bn_conv3_1)
    bn_conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization(momentum=0.01)(bn_conv3_2)
    bn_conv3_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(bn_conv3_3)

    #h * w = 32 * 32
    bn4 = BatchNormalization(momentum=0.01)(pool3)
    bn_conv4_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization(momentum=0.01)(bn_conv4_1)
    bn_conv4_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization(momentum=0.01)(bn_conv4_2)
    bn_conv4_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(bn_conv4_3)

    #h * w = 16 * 16
    bn5 = BatchNormalization(momentum=0.01)(pool4)
    bn_conv5_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn5)
    bn5 = BatchNormalization(momentum=0.01)(bn_conv5_1)
    bn_conv5_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn5)
    bn5 = BatchNormalization(momentum=0.01)(bn_conv5_2)
    bn_conv5_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn5)
    pool5 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(bn_conv5_3)


    #h * w = 8 * 8
    bn6 = BatchNormalization(momentum=0.01)(pool5)
    bn_conv6_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization(momentum=0.01)(bn_conv6_1)
    bn_conv6_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization(momentum=0.01)(bn_conv6_2)
    bn_upconv6 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2))(bn6)
    #Begin Deconvolution

    #h * w = 16 * 16
    concat7 = concatenate([bn_upconv6, bn_conv5_2], axis=3)
    bn7 = BatchNormalization(momentum=0.01)(concat7)
    bn_conv7_1 = Conv2D(96, (3, 3), activation='relu', padding='same')(bn7)
    bn7 = BatchNormalization(momentum=0.01)(bn_conv7_1)
    bn_conv7_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn7)
    bn7 = BatchNormalization(momentum=0.01)(bn_conv7_2)
    bn_upconv7 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2))(bn7)

    #h * w = 32 * 32
    concat8 = concatenate([bn_upconv7, bn_conv4_2], axis=3)
    bn8 = BatchNormalization(momentum=0.01)(concat8)
    bn_conv8_1 = Conv2D(96, (3, 3), activation='relu', padding='same')(bn8)
    bn8 = BatchNormalization(momentum=0.01)(bn_conv8_1)
    bn_conv8_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn8)
    bn8 = BatchNormalization(momentum=0.01)(bn_conv8_2)
    bn_upconv8 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2))(bn8)

    #h * w = 64 * 64
    concat9 = concatenate([bn_upconv8, bn_conv3_2], axis=3)
    bn9 = BatchNormalization(momentum=0.01)(concat9)
    bn_conv9_1 = Conv2D(96, (3, 3), activation='relu', padding='same')(bn9)
    bn9 = BatchNormalization(momentum=0.01)(bn_conv9_1)
    bn_conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn9)
    bn9 = BatchNormalization(momentum=0.01)(bn_conv9_2)
    bn_upconv9 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2))(bn9)

    #h * w = 128 * 128
    concat10 = concatenate([bn_upconv9, bn_conv2_1], axis=3)
    bn10 = BatchNormalization(momentum=0.01)(concat10)
    bn_conv10_1 = Conv2D(96, (3, 3), activation='relu', padding='same')(bn10)
    bn10 = BatchNormalization(momentum=0.01)(bn_conv10_1)
    bn_conv10_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn10)
    convout10 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn_conv10_2)

    #h * w = 128 * 128
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(convout10)

    model = Model(inputs=[inputs], outputs=[conv11])

    # model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy']) #modified by rizki.
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef,'binary_accuracy'])

    return model
