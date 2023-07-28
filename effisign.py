# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:12:24 2023

@author: Soundarya
"""
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers,Model
import matplotlib.image as mpimg
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
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

# Verify the sizes of the subsets
train_size = len(train_images)
test_size = len(test_images)
print(train_size, test_size)

# Normalize the pixel values of the images
train_images = train_images / 255.0
test_images = test_images / 255.0
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
num_of_classes=36
feature_extractor_layer = hub.KerasLayer(efficientnet_url,
                                           trainable=False, 
                                           name='feature_extraction_layer',
                                           input_shape=(224, 224, 3))
efficientnet_model = tf.keras.Sequential([
    feature_extractor_layer, 
    layers.Dense(num_of_classes, activation='softmax', name='output_layer')  
  ])

efficientnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])
from sklearn.preprocessing import OneHotEncoder

# One-hot encode the labels
encoder = OneHotEncoder()
train_labels_encoded = encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
test_labels_encoded = encoder.transform(test_labels.reshape(-1, 1)).toarray()
efficientnet_model.summary()

efficientnet_model_history = efficientnet_model.fit(train_images, train_labels_encoded,
                        epochs=35,
                        batch_size=64,
                        verbose=1,
                        validation_split=0.2)
efficientnet_model.evaluate(test_images, test_labels_encoded)
hist=efficientnet_model_history
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
predictions = efficientnet_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

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

# Generate the classification report
report = classification_report(test_labels, predicted_labels, target_names=class_names)
print(report)
