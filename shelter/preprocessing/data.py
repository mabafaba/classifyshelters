import os
import numpy as np

from skimage.io import imsave, imread

image_rows = 129
image_cols = 129


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def create_train_data(data_path):
    train_data_path = os.path.join(data_path, 'train')
    images = [path for path in os.listdir(train_data_path) if not path.startswith('Icon')]
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    print('Creating training images...')
    print('Dataset size:', len(images))

    for i, image_name in enumerate(images):
        if not any(x in image_name for x in ['mask', '.bmp']):
            image_mask_name = image_name.split('_sample.bmp')[0] + '_mask.bmp'
            img = imread(os.path.join(train_data_path, image_name), as_grey=True)
            img = np.array([img])
            # img = img.squeeze()

            img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
            img_mask = np.array([img_mask])

            imgs[i] = img
            imgs_mask[i] = img_mask

            # if i % 1000 == 0:
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

    np.save(os.path.join(data_path, 'imgs_train.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(data_path):
    imgs_train = np.load(os.path.join(data_path, 'imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(data_path, 'imgs_mask_train.npy'))

    return imgs_train, imgs_mask_train


def create_test_data(data_path):
    test_data_path = os.path.join(data_path, 'test')
    images = [path for path in os.listdir(test_data_path) if not path.startswith('Icon')]
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    print('Creating test images...')
    print('Dataset size:', len(images))

    for i, image_name in enumerate(images):
        if '.bmp' not in image_name:
            continue
        img_id = int(image_name.split('_')[0])
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
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
    np.save(os.path.join(data_path, 'imgs_test.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs_test = np.load(os.path.join(data_path, 'imgs_test.npy'))
    imgs_id = np.load(os.path.join(data_path, 'imgs_id_test.npy'))

    return imgs_test, imgs_id


if __name__ == '__main__':
    data_path = '/media/data'

    create_train_data(data_path)
    create_test_data(data_path)