import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize
import matplotlib.pyplot as plt

image_rows = 129
image_cols = 129


def create_train_data(data_path,showSample=False,showNumSample=1):
    train_data_path = os.path.join(data_path, 'input/train')
    images = [path for path in os.listdir(train_data_path) if not path.startswith('Icon')]

    sample_filename=[]
    mask_filename=[]
    for i, sample_name in enumerate(images):
        if 'sample' in sample_name and 'bmp' in sample_name:
            #loop again and only include if there is a corressponding mask file
            for j, mask_name in enumerate(images):
                if sample_name.replace('sample','mask')==mask_name:
                    sample_filename.append(sample_name)
                    mask_filename.append(mask_name)

    total = len(sample_filename)

    print('Creating training images...')
    print('Dataset size:', total)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    for i, image_name in enumerate(sample_filename):

        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img = np.array([img])

        img_mask = imread(os.path.join(train_data_path, image_name.replace('sample','mask')), as_gray=True)
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

    if showSample:
        print ('sample data:')
        plt.figure(figsize=(15,10))
        for i in range(1,showNumSample+1):
            plt.subplot(2,showNumSample,i)
            plt.imshow(imgs[i],cmap=('gray'))
            plt.subplot(2,showNumSample,i+showNumSample)
            plt.imshow(imgs_mask[i],cmap=('gray'))
        plt.show()

    print('Loading done.')

    if not os.path.exists(data_path+'/internal/npy'): os.makedirs(data_path+'/internal/npy')
    np.save(os.path.join(data_path, 'internal/npy/imgs_train.npy'), imgs)
    np.save(os.path.join(data_path, 'internal/npy/imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(data_path):
    imgs_train = np.load(os.path.join(data_path, 'internal/npy/imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(data_path, 'internal/npy/imgs_mask_train.npy'))

    return imgs_train, imgs_mask_train


def create_test_data(data_path):
    test_data_path = os.path.join(data_path, 'input/test')
    images = [path for path in os.listdir(test_data_path) if not path.startswith('Icon')]

    testSample_filename=[]
    for i, testSample_name in enumerate(images):
        if 'sample' in testSample_name and 'bmp' in testSample_name:
                testSample_filename.append(testSample_name)

    total = len(testSample_filename)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    print('Creating test images...')
    print('Dataset size:', total)

    for i, image_name in enumerate(testSample_filename):
        img_id = int(image_name.split('_')[0])
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        imgs_id[i] = img_id

    print('Loading done.')
    if not os.path.exists(data_path+'/internal/npy'): os.makedirs(data_path+'/internal/npy')
    np.save(os.path.join(data_path, 'internal/npy/imgs_test.npy'), imgs)
    np.save(os.path.join(data_path, 'internal/npy/imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs_test = np.load(os.path.join(data_path, 'internal/npy/imgs_test.npy'))
    imgs_id = np.load(os.path.join(data_path, 'internal/npy/imgs_id_test.npy'))
    return imgs_test, imgs_id


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) # reference:https://www.mathworks.com/matlabcentral/answers/234419-why-does-rgb2gray-use-these-weights-for-the-weighted-sum

def preprocess(imgs):
    # resize images
    resize_image_height_to = 128
    resize_image_width_to = 128

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

def normalize(imgs):

    # centering and normalise data
    imgs = imgs.astype('float32')
    mean = np.mean(imgs)  # mean for data centering
    std = np.std(imgs)  # std for data normalization
    imgs -= mean
    if std != 0:
        imgs /= std

    return imgs

def normalize_mask(imgs):

    # scale masks to [0, 1]
    imgs= imgs.astype('float32')
    imgs /= 255.  

    return imgs


if __name__ == '__main__':
    # data_path = '/media/data'

    create_train_data(data_path)
    create_test_data(data_path)
