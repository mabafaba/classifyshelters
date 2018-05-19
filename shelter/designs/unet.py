import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from preprocessing.data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# PARAMETERS
# data
resize_image_height_to = 128
resize_image_width_to  = 128
smooth                 = 1.0
test_data_fraction     = 0.15

# computation
number_of_epochs = 40
batch_size       = 80



# MODEL
# Loss function

# dice coefficient
def dice_coef(y_true, y_pred):
	# truth as vector:
	y_true_f = K.flatten(y_true)
	# prediction as vector:
	y_pred_f = K.flatten(y_pred)
	# count predicted "1"s that are also true "1"s = count true positives
	intersection = K.sum(y_true_f * y_pred_f)
	# 2 * count true positives / ( count true "1"s + count predicted "1"s
	# returns 0 for all wrong
	# returns 0 for prediction all 0
	# returns 1 for all correct
	# how 'good' a value in between depends is on total % of true "1"s.
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# loss is negative dice coefficient
def dice_coef_loss(y_true, y_pred):
	return - dice_coef(y_true, y_pred)


# Network architecture
def get_unet():

	# expected input shape
	inputs = Input((resize_image_height_to, resize_image_width_to, 1)) #  1 channel, x rows, y = x columns

	# convolution
	# Conv2D(number of filters, (kernel X, kernel Y), .. )
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) # -> convolution to  features: 32     window: 3
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  # -> convolution to  features: 32     window: 9
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                         # -> maxpool to      features: 32     image : x / ( 2 )

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # -> convolution to  features: 64     window: 18
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)  # -> convolution to  features: 64     window: 18
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                         # -> maxpool to      features: 64     image : x / (2^2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) # -> convolution to  features: 128    window: 54 
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) # -> convolution to  features: 128    window: 162
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                         # -> maxpool to      features: 128    image : x / (2^3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) # -> convolution to  features: 256    window: 486
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4) # -> convolution to  features: 256    window: 1458
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)                         # -> maxpool to      features: 256    window: ..

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4) # -> convolution to  features: 512    window: ..
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5) # -> convolution to  features: 512    image : x / (2^4)


	# deconvolution
	up6     = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5) # deconv to  features: 512    x / (2^4)
	concat6 = concatenate([up6, conv4], axis=3)                                   # add conv4
	conv6   = Conv2D(256, (3, 3), activation='relu', padding='same')(concat6)     # convolute
	conv6   = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)      # convolute

	up7     = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
	concat7 = concatenate([up7, conv3], axis=3)
	conv7   = Conv2D(128, (3, 3), activation='relu', padding='same')(concat7)
	conv7   = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8     = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8   = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8   = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# dont know why but i had commented this line.. which stopped loading existing weights (?)
	# aha: IF YOU HAVE NO WEIGHTS YET YOU NEED TO UNCOMMENT THIS LINE
	# when you run the the n>1th time copy the weights.h5 file from /output/ to /checkpoints/
	model.load_weights("weights.h5")
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

	return model


def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], resize_image_height_to, resize_image_width_to), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (resize_image_width_to, resize_image_height_to), preserve_range=True)

	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p


def train_and_predict():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()

	imgs_train = preprocess(imgs_train)
	imgs_mask_train = preprocess(imgs_mask_train)
	imgs_train = imgs_train.astype('float32')
	mean = np.mean(imgs_train)  # mean for data centering
	std = np.std(imgs_train)  # std for data normalization

	imgs_train -= mean
	if std!=0:
		imgs_train /= std


	imgs_mask_train = imgs_mask_train.astype('float32')
	imgs_mask_train /= 255.  # scale masks to [0, 1]
	print('mask max')

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = get_unet()

	model_checkpoint = ModelCheckpoint('/output/weights.h5', monitor='val_loss', save_best_only=True)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=number_of_epochs, verbose=1, shuffle=True,
			  validation_split=test_data_fraction,
			  callbacks=[model_checkpoint])

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
	# need to load newly trained weights from /output for prediction
	# even though i load last trainings weights from /checkpoint.
	# that's because floydhub only lets me write to /output/ during processing
	# because I have to write them to /output/
	#
	model.load_weights('/output/weights.h5')

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	imgs_mask_test = model.predict(imgs_test, verbose=1)
	np.save('/output/imgs_mask_test.npy', imgs_mask_test)

	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)
	pred_dir = '/output'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	for image, image_id in zip(imgs_mask_test, imgs_id_test):
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

	# for image, image_id in zip(imgs_train, imgs_id_test):
	#     image = (image[:, :, 0] * 255.).astype(np.uint8)
	#     imsave(os.path.join(pred_dir, str(image_id) + '_trainpred.png'), image)

if __name__ == '__main__':
	train_and_predict()
