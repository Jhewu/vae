""" 
This script stores the architecture of the 
Variational Autoencoder. Use with run_vae.py
"""

"""All imports"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
import tensorflow as tf
import keras
from keras import ops
from keras import layers

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
def Build_Encoder(IMAGE_SIZE, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM): 
    encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Conv2D(CONV_WIDTHS[0], 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for i in range(1, len(CONV_WIDTHS)): 
        x = layers.Conv2D(CONV_WIDTHS[i], CONV_KERNEL[i], strides=2, padding="same")(x) 
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if CONV_DEPTH > 0:
            for _ in range(CONV_DEPTH):
                x = layers.Conv2D(CONV_WIDTHS[i], CONV_KERNEL[i], strides=1, padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
    conv_z_mean = layers.Conv2D(LATENT_DIM, 3, strides=1, padding="same", name="conv_z_mean")(x)
    conv_z_log_var = layers.Conv2D(LATENT_DIM, 3, strides=1, padding="same", name="conv_z_log_var")(x)
    z = Sampling()([conv_z_mean, conv_z_log_var])
    return keras.Model(encoder_inputs, [conv_z_mean, conv_z_log_var, z], name="encoder")

"""Build the Decoder"""
def Build_Decoder(FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM): 
    latent_inputs = keras.Input(shape=(FINAL_OUTPUT.output.shape[1:]))
    print(latent_inputs)
    x = layers.Conv2D(LATENT_DIM, 3, strides=1, padding="same")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for i in range(len(CONV_WIDTHS)-1, -1, -1): 
        x = layers.Conv2DTranspose(CONV_WIDTHS[i], CONV_KERNEL[i], strides=2, padding="same")(x) 
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if CONV_DEPTH > 0:
            for _ in range(CONV_DEPTH):
                x = layers.Conv2DTranspose(CONV_WIDTHS[i], CONV_KERNEL[i], strides=1, padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

"""Build the VAE"""
class VAE(keras.Model):
    def __init__(self, IMAGE_SIZE, FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Build_Encoder(IMAGE_SIZE, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM)
        self.decoder = Build_Decoder(FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM)
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
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            kl_loss = tf.reduce_mean(kl_loss)
            # total loss
            total_loss = reconstruction_loss + kl_loss # * 100)
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
    def encode(self, image): 
        # returns z_mean, z_log_var, and z
        return self.encoder.predict(image)
    def decode(self, z): 
        # returns the decoded image
        return self.decoder.predict(z)