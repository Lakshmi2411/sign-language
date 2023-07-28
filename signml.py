import os
import tensorflow as tf
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import random as random
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
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
    img = load_img(image_path, target_size=(256, 256))  # Set the target size of the image
    row = i // 6
    col = i % 6
    axes[row][col].imshow(img)
    axes[row][col].set_title(class_name, fontsize=16)
    axes[row][col].axis('off')

# Show the figure
plt.show()

# Create lists to store the images and labels
images = []
labels = []

# Loop through each class and load the images and labels
for i, class_name in enumerate(class_names):
    class_path = os.path.join(dataset_dir, class_name)
    file_names = os.listdir(class_path)
    for file_name in file_names:
        image_path = os.path.join(class_path, file_name)
        img = load_img(image_path, target_size=(256, 256))
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

# Reshape the train_images array to 2D
train_images = train_images.reshape(train_size, -1)
test_images=test_images.reshape(test_size,-1)

# Normalize the pixel values of the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Support Vector Machine (SVM)
svm_model = svm.SVC()
svm_model.fit(train_images, train_labels)
svm_predictions = svm_model.predict(test_images)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_images, train_labels)
knn_predictions = knn_model.predict(test_images)

# Deep Belief Network (DBN)
dbn_model = BernoulliRBM(n_components=100)
dbn_model.fit(train_images)
dbn_features = dbn_model.transform(test_images)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(train_images, train_labels)
gb_predictions = gb_model.predict(test_images)

# Evaluate the models
print("SVM Classification Report:")
print(classification_report(test_labels, svm_predictions))
print("SVM Confusion Matrix:")
svm_cm = confusion_matrix(test_labels, svm_predictions)
sns.heatmap(svm_cm, annot=True, fmt="d")
plt.show()

print("KNN Classification Report:")
print(classification_report(test_labels, knn_predictions))
print("KNN Confusion Matrix:")
knn_cm = confusion_matrix(test_labels, knn_predictions)
sns.heatmap(knn_cm, annot=True, fmt="d")
plt.show()

print("Gradient Boosting Classification Report:")
print(classification_report(test_labels, gb_predictions))
print("Gradient Boosting Confusion Matrix:")
gb_cm = confusion_matrix(test_labels, gb_predictions)
sns.heatmap(gb_cm, annot=True, fmt="d")
plt.show()
