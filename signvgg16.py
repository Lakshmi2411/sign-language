# -*- coding: utf-8 -*-
"""
Created on Fri May 26 01:52:38 2023

@author: Soundarya
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
# define the model architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes=5
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(36, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random as random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import models, layers

data_dir = 'C:/Users/Soundarya/Downloads/sign language/asl_dataset'
class_names = os.listdir(data_dir)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    file_names = os.listdir(class_dir)
    num_files = len(file_names)
    print(f"Class {class_name} has {num_files} images")

dataset_dir = data_dir

# Get the list of class names from the directory
class_names = sorted(os.listdir(dataset_dir))

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))

# Loop through each class and plot one image from each class
for i, class_name in enumerate(class_names):
    class_path = os.path.join(dataset_dir, class_name)
    image_path = os.path.join(class_path, random.choice(os.listdir(class_path)))
    img = load_img(image_path, target_size=(224, 224))  # Set the target size of the image
    row = i // 6
    col = i % 6
    axes[row][col].imshow(img)
    axes[row][col].set_title(class_name, fontsize=16)
    axes[row][col].axis('off')

# Show the figure
plt.show()

from sklearn.model_selection import train_test_split

# Create lists to store the images and labels
images = []
labels = []
from tensorflow.keras.preprocessing.image import img_to_array
# Loop through each class and load the images and labels
for i, class_name in enumerate(class_names):
    class_path = os.path.join(dataset_dir, class_name)
    file_names = os.listdir(class_path)
    for file_name in file_names:
        image_path = os.path.join(class_path, file_name)
        img = load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(i)

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing subsets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels, num_classes=36)
test_labels = to_categorical(test_labels, num_classes=36)

# Verify the sizes of the subsets
train_size = len(train_images)
test_size = len(test_images)
print(train_size, test_size)

# Normalize the pixel values of the images
train_images = train_images / 255.0
test_images = test_images / 255.0
hist = model.fit(
    train_images,
    train_labels,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    epochs=45
)
print("Test accuracy and Test loss")

model.evaluate(test_images, test_labels)

# Plot the training and validation accuracy
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Make predictions on the test set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(test_labels.argmax(axis=1), predicted_labels)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)

# Set the labels and title
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# Rotate the tick labels for better visibility
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Show the plot
plt.show()

# Convert one-hot encoded test_labels back to original format
test_labels_original = np.argmax(test_labels, axis=1)

# Generate the classification report
report = classification_report(test_labels_original, predicted_labels, target_names=class_names)
print(report)

