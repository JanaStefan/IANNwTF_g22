# 1 // Prepare the Dataset

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

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

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
