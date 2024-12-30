"""
Implementation of Variational Autoencoders for
river images
"""

"""Import Libraries"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2 as cv
import datetime

"""Import Local"""
from vae_architecture import *

"""Hyperparameters"""
# general
SEED = 42
FOLDER_NAME = "flow_large"

# architecture
IMAGE_SIZE = (192, 592)
LATENT_DIM = 1024
CONV_WIDTHS = [64, 128, 256, 512]
CONV_KERNEL = [3, 3, 5, 5]
CONV_DEPTH = 1

# optimization
LEARNING_RATE = 3.5e-4
WEIGHT_DECAY = LEARNING_RATE/10
BATCH_SIZE = 8
EPOCHS = 100
DATASET_REPETITION = 1

# callbacks
CHECKPOINT_PATH = "checkpoints"
PATIENCE = 15
START_FROM_EPOCH = 0

# modes
MODE = "training"
LOAD_WEIGHTS = False
LOAD_WEIGHT_PATH = "checkpoints/best_2024-12-29 15:36:18.958169.weights.h5"
SAVE_IMAGE_SAMPLE = True
SAVE_IMAGE_SAMPLE_PATH = "sample_images"
NUM_IMAGES_TO_SAVE = 5

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def load_dataset(): 
    # load local dataset as tensorflow dataset object
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

def normalize_image(images, _):    
    # clip pixel values to the range [0, 1]
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def prepare_dataset(train_ds): 
    # use in combination with load_dataset()
    train_ds = (train_ds
                .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
                .cache()                                                   # (image, labels) when inputting to 
                .repeat(DATASET_REPETITION)                                # map
                .shuffle(10 * BATCH_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
    return train_ds

def show_images(training_dataset): 
    # showcase the first images in the dataset
    iterator = training_dataset.as_numpy_iterator()
    batches = next(iterator)
    img = batches[0]
    plt.imshow(img)
    plt.show()

def show_performance_visually(training_dataset, vae): 
    # demonstrate visual output of VAE

    # obtain the a single image
    iterator = training_dataset.as_numpy_iterator()
    batches = next(iterator)
    img = batches[0]

    # encode the batches
    #_, _, z = vae.encoder.predict(batches)
    _, _, z = vae.encode(batches)

    # decode from the encoder prediction
    #generated_sample = vae.decoder.predict(z)
    generated_sample = vae.decode(z)

    # create plots
    _, axes = plt.subplots(1, 3, figsize=(10, 10))

    decoded_image = generated_sample[0]

    axes[0].imshow(img)
    axes[1].imshow(decoded_image)
    plt.show()

def save_image_samples(training_dataset, vae, num_to_save, current_time): 
    # save image samples to review performance visually

    # obtain one batch
    training_dataset = training_dataset.shuffle(buffer_size=10)
    iterator = training_dataset.as_numpy_iterator()
    batches = next(iterator)

    # create destination directory
    root = os.getcwd()
    dest_dir = os.path.join(root, f"{SAVE_IMAGE_SAMPLE_PATH}/{current_time}")
    CreateDir(dest_dir)

    index = 0
    for image in batches[:num_to_save]: 
        exp_img = np.expand_dims(image, axis=0)
        print(exp_img.shape)

        # encode the batches
        #_, _, z = vae.encoder.predict(exp_img)
        _, _, z = vae.encode(exp_img)

        # decode from the encoder prediction
        #generated_sample = vae.decoder.predict(z)
        generated_sample = vae.decode(z)
        decoded_image = generated_sample[0]

        # scale both images
        scaled_img = image * 255
        scaled_generated_sample = decoded_image * 255

        # clip both images
        clipped_img = np.clip(scaled_img, 0, 255).astype(np.uint8)
        clipped_generated_sample = np.clip(scaled_generated_sample, 0, 255).astype(np.uint8)

        # get image directories
        original_image_name = os.path.join(dest_dir, f"original_img_{index}.jpg")
        decoded_image_name = os.path.join(dest_dir, f"decoded_img_{index}.jpg")

        # save both images
        cv.imwrite(original_image_name, clipped_img)
        cv.imwrite(decoded_image_name, clipped_generated_sample)
        index+=1

"""Main Runtimes"""
def RunVAE(current_time): 
    # Check if GPU tensorflow is using GPU
    print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

    # load and prepare the dataset
    dataset = load_dataset()
    training_dataset = prepare_dataset(dataset)

    # build the encoder to obtain the final output layer
    encoder = Build_Encoder(IMAGE_SIZE, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM)
    final_output = encoder.get_layer("conv_z_log_var")

    # build the VAE model
    vae = VAE(IMAGE_SIZE, final_output, CONV_WIDTHS, CONV_DEPTH, CONV_KERNEL, LATENT_DIM)

    if LOAD_WEIGHTS:
        vae.load_weights(LOAD_WEIGHT_PATH)

    if MODE == "training": 
        CreateDir(CHECKPOINT_PATH)

        # compile the model
        vae.compile(optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ))

        # Create a callback that saves the models' weights
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=f"{CHECKPOINT_PATH}/best_{current_time}.weights.h5",
                                                                save_best_only=True, 
                                                                save_weights_only=True, 
                                                                monitor="loss",
                                                                verbose=1,
                                                                mode="min")

        # Create an early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", 
                                                                min_delta=LEARNING_RATE*1000, 
                                                                mode="min",
                                                                patience=PATIENCE, 
                                                                verbose=1, 
                                                                restore_best_weights=True, 
                                                                start_from_epoch=START_FROM_EPOCH)

        # train the model
        vae.fit(x=training_dataset, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE,
                callbacks=[checkpoint_callback, early_stopping_callback])
        
        save_image_samples(training_dataset, vae, NUM_IMAGES_TO_SAVE, current_time)
        model.save_weights(f"{CHECKPOINT_PATH}/last_{current_time}.weights.h5")

    elif MODE == "inference": 
        save_image_samples(training_dataset, vae, NUM_IMAGES_TO_SAVE, current_time)
    else: 
        print("\nNo such MODE\nMODE is either 'training' or 'inference'\n")
    
if __name__ == "__main__": 
    # get current time
    current_time = datetime.datetime.now() 
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    RunVAE(formatted_time)
    print("\nFinish running VAE\n")
