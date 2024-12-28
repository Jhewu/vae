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
def Build_Encoder(IMAGE_SIZE, CONV_WIDTHS, CONV_DEPTH, KERNEL, LATENT_DIM): 
    """Potentially add block depth onto the model in the future to check performance"""
    encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Conv2D(CONV_WIDTHS[0], KERNEL, activation="relu", strides=2, padding="same")(encoder_inputs)
    for width in CONV_WIDTHS[1:]: 
        x = layers.Conv2D(width, KERNEL, activation="relu", strides=2, padding="same")(x) 
        if len(CONV_DEPTH) > 0:
            for depth in CONV_DEPTH:
                x = layers.Conv2D(depth, KERNEL, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    conv_z_mean = layers.Conv2D(LATENT_DIM, KERNEL, strides=1, padding="same", name="conv_z_mean")(x)
    conv_z_log_var = layers.Conv2D(LATENT_DIM, KERNEL, strides=1, padding="same", name="conv_z_log_var")(x)
    z = Sampling()([conv_z_mean, conv_z_log_var])
    return keras.Model(encoder_inputs, [conv_z_mean, conv_z_log_var, z], name="encoder")


"""Build the Decoder"""
def Build_Decoder(FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, KERNEL, LATENT_DIM): 
    """Potentially add block depth onto the model in the future to check performance"""
    latent_inputs = keras.Input(shape=(FINAL_OUTPUT.output.shape[1:]))
    print(latent_inputs)
    x = layers.Conv2D(LATENT_DIM, KERNEL, activation="relu", strides=1, padding="same")(latent_inputs)
    # add more convolutional layers here
    for width in CONV_WIDTHS[::-1]: 
        x = layers.Conv2DTranspose(width, KERNEL, activation="relu", strides=2, padding="same")(x) 
        if len(CONV_DEPTH) > 0:
            for depth in CONV_DEPTH:
                x = layers.Conv2DTranspose(depth, KERNEL, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(3, KERNEL, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

"""Build the VAE"""
class VAE(keras.Model):
    def __init__(self, IMAGE_SIZE, FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, KERNEL, LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Build_Encoder(IMAGE_SIZE, CONV_WIDTHS, CONV_DEPTH, KERNEL, LATENT_DIM)
        self.decoder = Build_Decoder(FINAL_OUTPUT, CONV_WIDTHS, CONV_DEPTH, KERNEL, LATENT_DIM)
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