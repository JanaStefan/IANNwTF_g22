# 1 // Prepare the Dataset

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, with_info=True)

# Define batch size
BATCH_SIZE = 64
PREFETCH_SIZE = 2

# Preprocess the dataset
def data_pipeline(input):
    # Map the dataset to extract images and labels
    input =input.map(lambda image, label: (image, label))
    # Reshape each image to a flat vector
    input = input.map(lambda image, label: (tf.reshape(image, (-1,)), label))
    # Normalize(Scale) image values to be in the range [-1, 1]
    input = input.map(lambda image, label: ((tf.cast(image, tf.float32) / 1023) - 1, label))
    # One-hot encode the labels
    input = input.map(lambda image, label: (image, tf.one_hot(label, depth=10)))
    # Decode one-hot labels and convert images to numpy arrays for visualization
    input = input.map(lambda image, label: (image, tf.argmax(label, axis=-1)))
    # Shuffle the dataset and create batches of size 4
    input = input.shuffle(1024).batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)
    return input

# Save the datasets after applying the data pipeline
train_dataset = data_pipeline(ds_train)
test_dataset = data_pipeline(ds_test)


# Plot the dataset
def plot_dataset(dataset, num_images=9):
    # Take a single batch from the dataset
    for images, labels in dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            # Reshape the image to 32x32x3
            img = images[i].numpy().reshape(32, 32, 3)
            # Rescale the image back to the range [0, 1] for visualization
            img = (img + 1) / 2
            plt.subplot(3, 3, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title("Label: {}".format(labels[i]))
            plt.axis('off')
        plt.show()

# Assuming we already define everything
plot_dataset(train_dataset)


