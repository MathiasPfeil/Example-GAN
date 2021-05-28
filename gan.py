from __future__ import print_function, division

import tensorflow as tf
import os
import cv2
import numpy as np

class DCGAN():
    def __init__(self):

        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 512


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.0001, 0.5),
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = tf.keras.models.Model(z, valid)
        self.combined.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001, 0.5))

    def build_generator(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(512 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(tf.keras.layers.Reshape((8, 8, 512)))
        model.add(tf.keras.layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding="same", activation=tf.nn.tanh))

        model.summary()

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return tf.keras.models.Model(noise, img)

    def build_discriminator(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", input_shape=self.img_shape, activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu, kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.sigmoid))

        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return tf.keras.models.Model(img, validity)

    def train(self, epochs, batch_size=64, save_interval=50):

        X_train = []

        path = os.path.join('./','assets/dataset/resized')

        for watch_img in os.listdir(path)[:25000]:
            bgr_img = cv2.imread(os.path.join(path, watch_img))
            ver_flipped = cv2.flip(bgr_img, 1)
            hor_flipped = cv2.flip(bgr_img, 0)
            ver_hor_flipped = cv2.flip(ver_flipped, 0)
            X_train.append(bgr_img)

        X_train = np.array(X_train)

        # Rescale -1 to 1
        X_train = X_train / 255

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.latent_dim]).astype(np.float32)

            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.uniform(-1.0, 1.0, size=[r * c, self.latent_dim]).astype(np.float32)
        gen_imgs = self.generator.predict(noise)

        # Terrible code... but I'm feeling lazy
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + '.jpg', ((gen_imgs[0] + 1) * 128).astype(int))
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + str(1) + '.jpg', ((gen_imgs[1] + 1) * 128).astype(int))
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + str(2) + '.jpg', ((gen_imgs[2] + 1) * 128).astype(int))
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + str(3) + '.jpg', ((gen_imgs[3] + 1) * 128).astype(int))
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + str(4) + '.jpg', ((gen_imgs[4] + 1) * 128).astype(int))
        cv2.imwrite("./assets/examples/gan_output/" + str(epoch) + str(5) + '.jpg', ((gen_imgs[5] + 1) * 128).astype(int))


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=100000, batch_size=32, save_interval=100)