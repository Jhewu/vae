"""Import Libraries"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers

import matplotlib.pyplot as plt

"""Check if its using GPU"""

print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

"""Hyperparameters"""

SEED = 42
FOLDER_NAME = "flow_large"
IMAGE_SIZE = (192, 576)
BATCH_SIZE = 2
EPOCHS = 100
DATASET_REPETITION = 1
LATENT_DIM = 512
CONV_WIDTHS = [32, 64, 128, 256, 512]
DENSE_WIDTHS = [512, 512]
KERNEL = 3
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 2.5e-5

"""Load the Dataset"""

def load_dataset(): 
    """
    Loads the dataset for training
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, FOLDER_NAME)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = None,
        subset=None, 
        seed = SEED,
        image_size = (IMAGE_SIZE[0], IMAGE_SIZE[1]),  
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = True,
        pad_to_aspect_ratio = False,
    )
    return train_ds

"""Prepare the Dataset"""

def normalize_image(images, _):    
    # clip pixel values to the range [0, 1]
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def prepare_dataset(train_ds): 
    """
    Prepares the dataset for training, used in combination with load_dataset
    """
    train_ds = (train_ds
                .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
                .cache()                                                   # (image, labels) when inputting to 
                .repeat(DATASET_REPETITION)                                # map
                .shuffle(10 * BATCH_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
    return train_ds

"""Load and Prepare Dataset and Show Sample Image"""

# load and prepare the dataset
dataset = load_dataset()
training_dataset = prepare_dataset(dataset)

# showcase the first image in the iterator
iterator = training_dataset.as_numpy_iterator()
batches = next(iterator)
img = batches[0]

#plt.imshow(img)
#plt.show()

"""Sampling layer"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=tf.shape(z_log_var), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

"""Build the Encoder"""

def Build_Encoder(): 
    """Potentially add block depth onto the model in the future to check performance"""
    encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Conv2D(CONV_WIDTHS[0], KERNEL, activation="relu", strides=2, padding="same")(encoder_inputs)
    for width in CONV_WIDTHS[1:]: 
        x = layers.Conv2D(width, KERNEL, activation="relu", strides=2, padding="same")(x) 
        x = layers.Conv2D(width, KERNEL, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    conv_z_mean = layers.Conv2D(LATENT_DIM, KERNEL, strides=1, padding="same", name="conv_z_mean")(x)
    conv_z_log_var = layers.Conv2D(LATENT_DIM, KERNEL, strides=1, padding="same", name="conv_z_log_var")(x)
    z = Sampling()([conv_z_mean, conv_z_log_var])
    return keras.Model(encoder_inputs, [conv_z_mean, conv_z_log_var, z], name="encoder")

encoder = Build_Encoder()
encoder.summary()

conv_z_log_var = encoder.get_layer("conv_z_log_var")

print(conv_z_log_var.output.shape[1:])

"""Build the Decoder"""

from math import prod

# need to manually set this value if you 
# change model architecture
final_shape = (6, 18, 512)

def Build_Decoder(): 
    """Potentially add block depth onto the model in the future to check performance"""
    latent_inputs = keras.Input(shape=(conv_z_log_var.output.shape[1:]))
    print(latent_inputs)
    x = layers.Conv2D(LATENT_DIM, KERNEL, activation="relu", strides=1, padding="same")(latent_inputs)
    # add more convolutional layers here
    for width in CONV_WIDTHS[::-1]: 
        x = layers.Conv2DTranspose(width, KERNEL, activation="relu", strides=2, padding="same")(x) 
        x = layers.Conv2DTranspose(width, KERNEL, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(3, KERNEL, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder = Build_Decoder()
decoder.summary()

"""Build the VAE"""

class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Build_Encoder()
        self.decoder = Build_Decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        print("passed")
        with tf.GradientTape() as tape:
            print("passed")
            z_mean, z_log_var, z = self.encoder(data)
            print("passed")
            reconstruction = self.decoder(z)
            print(reconstruction)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

"""Train the VAE"""

CHECKPOINT_PATH = "checkpoints/best.weights.h5"

vae = VAE()
vae.compile(optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
))

# Create a callback that saves teh models' weights
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                                                         save_best_only=True, 
                                                         save_weights_only=True, 
                                                         monitor=vae.kl_loss_tracker,
                                                         verbose=1,
                                                         mode="min")

vae.fit(x=training_dataset, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback])

"""Demonstrate Visual Output/Performance of VAE"""

import matplotlib.pyplot as plt

# inference
# latent_dim = 2
# z_sample = np.random.normal(size=(1, latent_dim))
# generated_sample = vae.decoder.predict(z_sample)

# get img
iterator = training_dataset.as_numpy_iterator()
batches = next(iterator)
img = batches[0]

print(img.shape)

# encoder
z_mean, z_log_var, z = vae.encoder.predict(batches)

# Remove the first dimension 
z_sliced = z[0] 

print(z_sliced.shape)

# decoder
generated_sample = vae.decoder.predict(z)

# create plots
fig, axes = plt.subplots(1, 3, figsize=(10, 10))

# show img
axes[0].imshow(img)
#axes[1].imshow(z[0])
axes[2].imshow(generated_sample[0])

#plt.imshow(generated_sample[0])
#plt.show()

"""Save the Image for Better Visualization"""

import cv2 as cv
import time

current_time = time.time()

scaled_img = img * 255
scaled_generated_sample = generated_sample[0] * 255

clipped_img = np.clip(scaled_img, 0, 255).astype(np.uint8)
clipped_generated_sample = np.clip(scaled_generated_sample, 0, 255).astype(np.uint8)

cv.imwrite(f"original_img{current_time}.jpg", clipped_img)
cv.imwrite(f"decoder_img{current_time}.jpg", clipped_generated_sample)

print(np.min(generated_sample[0]))
print(np.max(generated_sample[0]))

"""Save the Weights"""

vae.save_weights(f'vae_large{current_time}.weights.h5')