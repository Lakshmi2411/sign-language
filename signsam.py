# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:46:36 2023

@author: Soundarya
"""

import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
from tensorflow.keras import models, layers

data_dir = 'C:/Users/Soundarya/Downloads/sign language/asl_dataset'
class_names = os.listdir(data_dir)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    file_names = os.listdir(class_dir)
    num_files = len(file_names)
    print(f"Class {class_name} has {num_files} images")
import random as random

dataset_dir = data_dir

# Get the list of class names from the directory
class_names = sorted(os.listdir(dataset_dir))

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))

# Loop through each class and plot one image from each class
for i, class_name in enumerate(class_names):
    class_path = os.path.join(dataset_dir, class_name)
    image_path = os.path.join(class_path, random.choice(os.listdir(class_path)))
    img = load_img(image_path, target_size=(224, 224)) # Set the target size of the image
    row = i // 6
    col = i % 6
    axes[row][col].imshow(img)
    axes[row][col].set_title(class_name, fontsize=16)
    axes[row][col].axis('off')

# Show the figure
plt.show()

def get_dataset(ds, train_split=0.6, val_split = 0.2, test_split = 0.2, shuffle = True, shuffle_size = 1000):
    assert(train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    if shuffle:
        df = ds.shuffle(shuffle_size, seed = 12)
        
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
#     test_size = int(test_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed = 99,
    shuffle = True,
    image_size = (256,256),
    batch_size = 64
) 
train_ds, val_ds, test_ds = get_dataset(dataset)

model = models.Sequential([
    layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(36, activation='softmax'),
])
model.summary()
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)
hist = model.fit(
    train_ds,
    batch_size = 64,
    validation_data = val_ds,
    verbose = 1,
    epochs = 35,
    
)
model.evaluate(test_ds)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


