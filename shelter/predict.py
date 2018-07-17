import os
import numpy as np

from skimage.io import imsave
from keras import backend as K
from shelter.preprocessing.data import load_train_data, load_test_data
K.set_image_data_format('channels_last')  # TF dimension ordering in this code



def predict(data_path,model_str):

    #add models here:
    if model_str=='unet':from shelter.designs import unet as design
    if model_str=='unet64filters':from shelter.designs import unet64filters as design
    if model_str=='flatunet':from shelter.designs import flatunet as design
    if model_str=='unet64batchnorm':from shelter.designs import unet64batchnorm as design

    # input_path = os.path.join(data_path, 'input')
    imgs_train, imgs_mask_train = load_train_data(data_path)
    print(len(imgs_train))

    imgs_train = design.preprocess(imgs_train)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    print('Creating and compiling model...')
    model = design.build()
   
    print('Loading and preprocessing test data...')

    imgs_test, imgs_id_test = load_test_data(data_path)
    imgs_test = design.preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    ckpt_path = os.path.join(data_path, 'internal/checkpoints')
    ckpt_file = os.path.join(ckpt_path, 'weights_'+model_str+'.h5')
    #ckpt_file = os.path.join(ckpt_path, 'weights_file.h5') #or enter specific weights file here.
    model.load_weights(ckpt_file)
    print('Loading saved weights :',ckpt_file)

    print('Predicting masks on test data...')
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    out_path = os.path.join(data_path, 'output')
    out_file = os.path.join(data_path, 'internal/npy/imgs_mask_test.npy')
    np.save(out_file, imgs_mask_test)

    print('Saving predicted masks to files...')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(out_path, '{0:0>5}_pred.png'.format(image_id)), image)

    # for image, image_id in zip(imgs_train, imgs_id_test):
    #     image = (image[:, :, 0] * 255.).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_trainpred.png'), image)


if __name__ == '__main__':
    data_path = '/media/data/180505_v1'
    predict(data_path)
