import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization, Reshape, Dense, Input, LeakyReLU, Dropout, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import os
from utils import *
from glob import glob
import numpy as np



class DCGAN(object):

    def __init__(self, **kwargs):
        self.model_name = "DCGAN"
        self.dataset_name = kwargs.get('dataset')
        self.checkpoint_dir = kwargs.get('checkpoint_dir')
        self.sample_dir = kwargs.get('sample_dir')
        self.result_dir = kwargs.get('result_dir')
        self.log_dir = kwargs.get('log_dir')

        self.epoch = kwargs.get('epoch', 10)
        self.iteration = kwargs.get('iteration', 10000)
        self.batch_size = kwargs.get('batch_size', 16)
        self.print_freq = kwargs.get('print_freq', 500)
        self.save_freq = kwargs.get('save_freq', 500)
        self.img_size = kwargs.get('img_size', 128)
        self.sample_num = kwargs.get('sample_num', 64)

        """ Augmentation """
        self.crop_pos = kwargs.get('crop_pos', 'center')
        self.rotation_range = kwargs.get('rotation_range', 0)
        self.zoom_range = kwargs.get('zoom_range', 0.0)


        self.upsample_layer = 5
        self.starting_filters = 64
        self.kernel_size = 3
        self.c_dim = 3
        self.z_dim = kwargs.get('z_dim', 128)  # dimension of noise-vector

        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.beta1 = kwargs.get('beta1', 0.0)
        self.beta2 = kwargs.get('beta2', 0.9)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.train_generator = load_data(self.dataset_name, size=self.img_size,
                               rotation=self.rotation_range,
                               crop_pos=self.crop_pos,
                               zoom_range=self.zoom_range,
                               batch_size=self.batch_size)
        self.generator = None
        self.discriminator = None
        self.gan = None

        self.start_epoch = None
        self.start_batch_id = None
        self.counter = None


    def build_generator(self):
        model = Sequential()
        d = self.img_size // (2 ** self.upsample_layer) # 256/2^5 = 4

        # 8x8x64
        model.add(Dense(self.starting_filters * d * d, activation='relu', input_shape=(self.z_dim,)))
        model.add(Reshape((d, d, self.starting_filters)))
        model.add(BatchNormalization(momentum=0.8))

        # 8x8 -> 16x16
        model.add(UpSampling2D())
        model.add(Conv2D(1024, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        # 16x16 -> 32x32
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        # 32x32 -> 64x64
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        # 64x64 -> 128x128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        # 128x128 -> 256x256
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.c_dim, kernel_size=self.kernel_size, padding='same'))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.z_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = (self.img_size, self.img_size, self.c_dim)
        model = Sequential()

        #256x256 -> 128x128
        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 128x128 -> 64x64
        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        # 64x64 -> 32x32
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        # 32x32 -> 16x16
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 16x16 -> 8x8
        model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_models(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        g_models = sorted(glob(f'{checkpoint_dir}/generator*.h5'))
        d_models = sorted(glob(f'{checkpoint_dir}/discriminator*.h5'))
        model_names = []
        if len(g_models) > 0:
            for i in range(len(g_models)):
                g_model = g_models[i]
                d_model = d_models[i]
                model_name = '-'.join(('.'.join(os.path.basename(g_model).split('.')[:-1])).split('-')[1:])
                model_names.append(model_name)
                print(f'[{i}] {model_name}')
            selected_index = -1
            while not selected_index in range(i+1):
                try:
                    selected_index = int(input('select model to load: '))
                except:
                    print('select the model with model index')
            print(model_names[selected_index])
            d_path = d_models[selected_index]
            g_path = g_models[selected_index]
            print(d_path)
            print(g_path)
            self.discriminator = load_model(d_path)
            self.generator = load_model(g_path)
            counter = int(next(re.finditer("(\d+)(?!.*\d)",model_names[selected_index])).group(0))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def build_gan(self):
        # optimizer = Adam(self.learning_rate, beta_1=self.beta1, beta_2=self.beta2)
        optimizer = Adam(self.learning_rate, 0.5)

        could_load, checkpoint_counter = self.load_models(self.checkpoint_dir)

        if could_load:
            self.start_epoch = (int)(checkpoint_counter / self.iteration)
            self.start_batch_id = checkpoint_counter - self.start_epoch * self.iteration
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.discriminator.trainable = True
            self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.counter = checkpoint_counter
            print("[*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.start_epoch = 0
            self.start_batch_id = 0
            self.counter = 1

        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.gan = Model(z, valid)
        self.gan.summary()
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)



    def train(self, z_noise=None):
        start_time = time.time()

        # log writer
        logdir = self.log_dir + '/' + self.model_dir
        tensorboard_callback = TensorBoard(log_dir=logdir)
        tensorboard_callback.set_model(self.gan)
        scalar_names = ['d_loss', 'g_loss']


        for epoch in range(self.start_epoch, self.epoch):
            # get batch data
            for idx in range(self.start_batch_id, self.iteration):

                # train generator
                noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
                y_gen = np.ones(self.batch_size)
                g_loss = self.gan.train_on_batch(noise, y_gen)


                # train discriminator
                half_batch = self.batch_size // 2
                real_x, _ = next(self.train_generator)
                while real_x.shape[0] != half_batch:
                    real_x, _ = next(train_generator)
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                fake_x = self.generator.predict(noise)
                real_y = np.ones(half_batch)
                real_y[:] = 0.9
                fake_y = np.zeros(half_batch)
                d_loss_real = self.discriminator.train_on_batch(real_x, real_y)
                d_loss_fake = self.discriminator.train_on_batch(fake_x, fake_y)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



                # write summary
                self.write_log(tensorboard_callback, scalar_names, [d_loss, g_loss], self.counter)

                # display training status
                self.counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # save training results for every n steps
                if type(z_noise) is np.ndarray:
                    sample_num = z_noise.shape[0]
                else:
                    sample_num = self.sample_num
                    z_noise = np.random.normal(0, 1, (sample_num, self.z_dim))

                if np.mod(idx+1, self.print_freq) == 0:
                    sample_imgs = self.generator.predict(z_noise)
                    manifold = int(np.ceil(np.sqrt(sample_num)))
                    save_images_plt(sample_imgs, [manifold, manifold],
                    f'{self.sample_dir}/{self.model_name}_train_{epoch:02d}_{idx+1:05d}', mode='sample')

                if np.mod(idx+1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir,  self.counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir,  self.counter)

        # save model for the final step
        self.save(self.checkpoint_dir,  self.counter)


    @property
    def model_dir(self):
        dataset_name = self.dataset_name.split('/')[-1]
        return "{}_{}_{}_{}".format(
            self.model_name, dataset_name, self.img_size, self.z_dim)

    def get_images_iterator(self):
        return next(self.train_generator)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        g_name = f'generator-{self.model_name}-{step:05d}.h5'
        d_name = f'discriminator-{self.model_name}-{step:05d}.h5'
        g_path = f'{checkpoint_dir}/{g_name}'
        d_path = f'{checkpoint_dir}/{d_name}'
        self.generator.save(g_path, include_optimizer=False)
        self.discriminator.save(d_path, include_optimizer=False)


    def write_log(self, callback, names, logs, step):
        for name, value in zip(names, logs):
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, step)
            callback.writer.flush()
