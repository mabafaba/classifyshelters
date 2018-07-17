import os
import numpy as np

from skimage.io import imsave, imread

image_rows = 129
image_cols = 129


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def create_train_data(data_path):
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
    print('imgs.shape =',imgs.shape)
    print('imgs_mask.shape =',imgs_mask.shape)

    for i, image_name in enumerate(sample_filename):

        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img = np.array([img])
        # img = img.squeeze()

        img_mask = imread(os.path.join(train_data_path, image_name.replace('sample','mask')), as_gray=True)
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        # if i % 100 == 0:
        #     print('Done: {0}/{1} images'.format(i, total))
        #     print("samp data type:")
        #     print(img.dtype)
        #     print("mask data type ")
        #     print(img_mask.dtype)
        #     print("samp shape ")
        #     print(img.shape)
        #     print("mask shape ")
        #     print(img_mask.shape)

    print('Loading done.')

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

        # if i % 300 == 0:
        #     print('Done: {0}/{1} images'.format(i, total))
        #     print("samp data type:")
        #     print(img.dtype)
        #     print("samp shape ")
        #     print(img.shape)

    print('Loading done.')
    np.save(os.path.join(data_path, 'internal/npy/imgs_test.npy'), imgs)
    np.save(os.path.join(data_path, 'internal/npy/imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs_test = np.load(os.path.join(data_path, 'internal/npy/imgs_test.npy'))
    imgs_id = np.load(os.path.join(data_path, 'internal/npy/imgs_id_test.npy'))
    return imgs_test, imgs_id


if __name__ == '__main__':
    # data_path = '/media/data'

    create_train_data(data_path)
    create_test_data(data_path)
