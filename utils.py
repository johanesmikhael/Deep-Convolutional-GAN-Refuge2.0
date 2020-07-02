import os
import pickle
import sys
import preprocess_crop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
import time
import matplotlib.pyplot as plt
import imageio
import numpy as np



def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def write_param(param_txt_path, param_pickle_path, kwargs):
    with open(param_pickle_path, 'wb') as f:
        pickle.dump(kwargs, f)
    with open(param_txt_path, 'w') as f:
        for key, value in kwargs.items():
            f.write(f'{key} : {value}\n')

def print_param(kwargs):
    for key, val in kwargs.items():
        print(f'{key} \t: {val}')

def check_param(param, param_pickle, kwargs):
    if os.path.isfile(param_pickle):
        with open(param_pickle, 'rb') as f:
            saved_kwargs = pickle.load(f)
        diff = {}
        for key, value in kwargs.items():
            if saved_kwargs.get(key) != value:
                diff[key] = [saved_kwargs.get(key, None), value]
        if diff:
            print('The previous parameter is different with the new ones:')
            print('------------')
            for key, value in diff.items():
                print(f'{key} \t: old value = {value[0]}, new value = {value[1]}')
            print('------------')
            option = ''
            while not option in ['p', 'P', 'n', 'N', 'a', 'A']:
                try:
                    print('Select an option:')
                    print('[P] to continue with the previous param')
                    print('[N] to continue with the new param')
                    print('[A] to abort operation')
                    option = str(input('type [P/N/A] and hit enter : '))
                except:
                    pass
            if option in ['p', 'P']:
                print('continue with the previous param')
                kwargs = saved_kwargs
            if option in ['n', 'N']:
                print('continue with new param')
                write_param(param, param_pickle, kwargs)
            if option in ['a', 'A']:
                print('aborting operation restart the runtime and run from the beginning')
                sys.exit()
    else:
        write_param(param, param_pickle, kwargs)
        print('model parameters is saved')

    print_param(kwargs)

    return kwargs

def load_data(dataset_name, size=64, rotation=0, crop_pos='center', zoom_range=0.0, batch_size=16):
    half_batch = batch_size // 2
    classes = [dataset_name.split('/')[-1]]
    path = '/'.join(dataset_name.split('/')[:-1])
    print(path)
    if os.path.isdir(path):
        print('dataset is exist...')
    train_datagen = ImageDataGenerator(
        rotation_range=rotation,
		    zoom_range=0.15,
        horizontal_flip=True,
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=path,
        target_size=(size, size),
        color_mode="rgb",
        batch_size=half_batch,
        classes = classes,
        class_mode="categorical",
        interpolation = f'lanczos:{crop_pos}',
        shuffle=True,
    )
    return train_generator


def save_images_plt(images, size, image_path, mode=None):
    images = inverse_transform(images)
    images = to_uint8(images)
    if mode == 'sample':
        h = 10
    else:
        h = 21.6
        img_dir = '/'.join(image_path.split('/')[:-1])+'/'+image_path.split('/')[-1][:-4]
        print(img_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
    w = size[0]/size[1] * h
    plt.figure(figsize=(w,h), dpi=100)
    n_rows = size[1]
    n_cols = size[0]

    for i in range(images.shape[0]):
        plt.subplot(n_rows, n_cols, i+1)
        image = images[i]
        if mode != 'sample':
            img_path = f'{img_dir}/{i:03d}.png'
            imageio.imwrite(img_path, image)
        if image.shape[2] == 1:
            plt.imshow(image.reshape((image.shape[0], image.shape[1])), cmap='gray')
        else:
            plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    is_exist = os.path.isfile(image_path)
    i = 1
    image_path_temp = image_path
    while is_exist == True:
        image_path = image_path_temp[:-4] + f' ({i:02d})'+image_path_temp[-4:]
        is_exist = os.path.isfile(image_path)
        i+=1
    plt.savefig(image_path)
    plt.close()

def save_image(image, image_path):
    image = inverse_transform(image)
    image = to_uint8(image)
    imageio.imwrite(image_path, image)

def inverse_transform(images):
    return (images+1.)/2.

def to_uint8(images):
    return (images * 255).astype(np.uint8)
