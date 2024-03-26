
import os.path
from time import sleep

import cv2 as cv
import numpy as np
import glob
import pandas as pd
from skimage.feature import hog
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras

def show_image(title, photo):
    cv.imshow(title, photo)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_bounding_boxes(file_path):
    columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'character']
    df = pd.read_csv(file_path, sep=' ', header=None, names=columns)
    return df

def prepare_data(all_photos, all_annotations):
    x_train = []

    input_shape = (96, 96)
    for character_photos, character_annotations in zip(all_photos, all_annotations):
        for photo, row in zip(character_photos, character_annotations.itertuples(index=False)):
            _, x_min, y_min, x_max, y_max, _ = row
            face = photo[y_min:y_max, x_min:x_max]
            # print(f"shape of face {face.shape}")
            face = cv.resize(face, (input_shape[1], input_shape[0]))  # Resize to model input size
            x_train.append(face)
            # y_train.append([x_min, y_min, x_max, y_max])  # You might need to normalize these values

    # x_train = np.array(x_train) / 255.0  # Normalize pixel values
    print(len(x_train))
    return x_train

def draw_bounding_box(image, x_min, y_min, x_max, y_max, color = (0, 255, 0), thickness = 2):
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def prepare_show_data(all_photos, all_annotations):
    n = len(all_annotations)
    x_train = []
    # y_train = []
    input_shape = (96, 96)
    for i in range(n):
        for index, row in all_annotations[i].iterrows():
            # You can access each column's value using the row variable.
            # For example, row['image_name'] will give you the image name.
            # print(f"Index: {index}, Image Name: {row['image_name']}, x_min: {row['x_min']}, y_min: {row['y_min']}, x_max: {row['x_max']}, y_max: {row['y_max']}, Character: {row['character']}")
            nr_image = int(row['image_name'].replace('.jpg', ''))
            current_photo = all_photos[i][nr_image - 1]
            # show_image("current_photo", current_photo)
            # current_photo_with_box = current_photo
            # image_with_box = draw_bounding_box(current_photo_with_box, row['x_min'], row['y_min'], row['x_max'], row['y_max'])
            # show_image("Image with box", image_with_box)
            y_min = row['y_min']
            y_max = row['y_max']
            x_min = row['x_min']
            x_max = row['x_max']
            face = current_photo[y_min:y_max, x_min:x_max]

            # print(f"shape of face {face.shape}")
            face = cv.resize(face, (input_shape[1], input_shape[0]))  # Resize to model input size
            x_train.append(face)
            # y_train.append([x_min, y_min, x_max, y_max])  # You might need to normalize these values

    # x_train = np.array(x_train) / 255.0  # Normalize pixel values
    # y_train = np.array(y_train)
    print("cate fete sunt")
    print(len(x_train))
    return x_train

def read_data():
    # /kaggle/working/
    train_path = os.path.join("CAVA-2023-TEMA2", "antrenare")
    # train_path = "/content/drive/MyDrive/CAVA-2023-TEMA2/antrenare"
    names = ["barney", "betty", "fred", "wilma"]
    all_photos = []
    all_annotations = []
    for name in names:
        current_character = []
        current_path = os.path.join(train_path, name, "*.jpg")
        jpg_files = sorted(glob.glob(current_path))  # Sort the file paths

        for file_path in jpg_files:
            image = cv.imread(file_path)
            if image is not None:
                current_character.append(image)
            else:
                print(f"Error reading image at path: {file_path}")

        all_photos.append(current_character)
        annotations = read_bounding_boxes(os.path.join(train_path, name + "_annotations.txt"))
        all_annotations.append(annotations)

    return all_photos, all_annotations

def read_validation_data():
    path = os.path.join("CAVA-2023-TEMA2", "validare", "validare")
    # path = "/content/drive/MyDrive/CAVA-2023-TEMA2/validare/validare"
    current_path = os.path.join(path, "*.jpg")
    jpg_files = sorted(glob.glob(current_path))  # Sort the file paths
    validation_photos = []
    for file_path in jpg_files:
        image = cv.imread(file_path)
        if image is not None:
            validation_photos.append(image)
        else:
            print(f"Error reading image at path: {file_path}")
    path = os.path.join("CAVA-2023-TEMA2", "validare", "validare_annotations.txt")
    # path = "/content/drive/MyDrive/CAVA-2023-TEMA2/validare/validare_annotations.txt"
    df = read_bounding_boxes(path)
    return validation_photos, df

def variable_sliding_window(image, minSize, maxSize, stepSize, scale=1.2):
    """
    Slides variable-sized windows across the image.
    Args:
    - image (numpy array): The image over which to slide the window.
    - minSize (tuple of int): The minimum width and height of the window (w, h).
    - maxSize (tuple of int): The maximum width and height of the window (w, h).
    - stepSize (int): The number of pixels to skip in both the x and y direction.
    - scale (float): The scale factor to increase the window size in each iteration.
    """
    # Start with the minimum size window
    windowSize = minSize
    while windowSize[0] <= maxSize[0] and windowSize[1] <= maxSize[1]:
        for y in range(0, image.shape[0] - windowSize[1], stepSize):
            for x in range(0, image.shape[1] - windowSize[0], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

        # Increase the window size
        windowSize = (int(windowSize[0] * scale), int(windowSize[1] * scale))

def extract_negative_patches(image, face_coords, patch_size, num_patches=2, margin=10, max_attempts=100):
    negative_patches = []
    img_height, img_width = image.shape[:2]
    patch_width, patch_height = patch_size

    for _ in range(num_patches):
        attempts = 0
        while attempts < max_attempts:
            x_min = random.randint(0, img_width - patch_width)
            y_min = random.randint(0, img_height - patch_height)
            x_max = x_min + patch_width
            y_max = y_min + patch_height

            overlap = any((x_min < fx_max + margin) and (x_max + margin > fx_min) and
                          (y_min < fy_max + margin) and (y_max + margin > fy_min)
                          for fx_min, fy_min, fx_max, fy_max in face_coords)

            if not overlap:
                patch = image[y_min:y_max, x_min:x_max]
                negative_patches.append(patch)
                break

            attempts += 1

    return negative_patches

def get_outsider_negatives(num_photos):
    current_path = os.path.join("exempleNegative", "*.jpg")
    # current_path = "/content/drive/MyDrive/exempleNegative"
    current_path = os.path.join(current_path, "*.jpg")
    jpg_files = sorted(glob.glob(current_path))  # Sort the file paths
    imgs = []
    for i in range(len(jpg_files)):
        image = cv.imread(jpg_files[i])
        image = cv.resize(image, (96, 96))
        if image is not None:
            imgs.append(image)
    return imgs

def intersection_over_union( bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou
def non_maximal_suppression( image_detections, image_scores, image_size):
    """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
    """

    # xmin, ymin, xmax, ymax
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

def is_skin_color_present(face_image):
    # Convert the image to the YCbCr color space
    ycbcr_image = cv.cvtColor(face_image, cv.COLOR_BGR2YCrCb)

    # Define the range for skin color in YCbCr
    min_YCrCb = np.array([50, 73, 70], np.uint8)
    max_YCrCb = np.array([225, 221, 202], np.uint8)

    # Create a mask to detect skin color
    skin_mask = cv.inRange(ycbcr_image, min_YCrCb, max_YCrCb)

    # Calculate the percentage of skin pixels in the face region
    skin_percentage = np.sum(skin_mask == 255) / (skin_mask.size) * 100

    return skin_percentage > 50  # Return True if more than 50% of the face region contains skin color

class FaceClassifier(tf.keras.Model):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.maxpool1 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

    def get_config(self):
        config = super(FaceClassifier, self).get_config()
        return config

"""MAIN"""



photos, annotations = read_data()
print("Data is read")
# Assuming the first image represents the common shape
first_image_shape = photos[0][0].shape
input_shape = (first_image_shape[0], first_image_shape[1], 3)  # (Height, Width, Channels)
print(first_image_shape)

# OBTAIN THE POSITIVE
x_train = prepare_show_data(photos, annotations)
# prepare_data(photos, annotations)
print("Data is prepared for training")

# create negative data
# Initialize a dictionary to store face coordinates for each image
image_face_coords = {}
n = len(annotations)

for i in range(n):
    for index, row in annotations[i].iterrows():
        image_name = row['image_name']
        face_coords = (row['x_min'], row['y_min'], row['x_max'], row['y_max'])

        if image_name not in image_face_coords:
            image_face_coords[image_name] = [face_coords]
        else:
            image_face_coords[image_name].append(face_coords)

negative_patches = []
for i in range(n):
    for index, row in annotations[i].iterrows():
        nr_image = int(row['image_name'].replace('.jpg', ''))
        current_photo = photos[i][nr_image - 1]
        current_face_coords = image_face_coords[row['image_name']]
        neg_vals = extract_negative_patches(current_photo, current_face_coords, patch_size=(96, 96), num_patches=1)
        if neg_vals:
            negative_patches.extend(neg_vals)

negative2_path = "train/"
train_negative = []
for elem in os.listdir(negative2_path):
    file_path = os.path.join(negative2_path, elem)
    image = cv.imread(file_path)
    if image is not None:
        resized_image = cv.resize(image, (96, 96))
        train_negative.append(resized_image)

negative_images = get_outsider_negatives(num_photos=274)
positive_labels = np.ones(len(x_train))
negative_labels = np.zeros(len(negative_patches) + len(negative_images) + len(train_negative))

all_negative_samples = negative_patches + negative_images + train_negative
x_train = np.array(x_train)
flipped_data = [np.fliplr(face) for face in x_train]
x_train = x_train + flipped_data

x_train = np.array(x_train)
print(f"x_train.shape = {x_train.shape}")

all_negative_samples = np.array(all_negative_samples)
print(f"all_negative_samples.shape = {all_negative_samples.shape}")

all_data = np.concatenate([x_train, all_negative_samples], axis=0)
all_labels = np.concatenate([positive_labels, negative_labels], axis=0)
all_data = np.array(all_data) / 255.0
all_labels = np.array(all_labels)

indices = np.arange(all_data.shape[0])
np.random.shuffle(indices)
shuffled_train_data = all_data[indices]
shuffled_train_labels = all_labels[indices]

# CITIRE DATE VALIDARE
validation_photos, val_annotations = read_validation_data()

# ####################### VERIFICARE DACA ESTE SALVAT DEJA MODELUL ######
# model_path = '/content/drive/MyDrive/cnn_saved_model.h5'
# if not os.path.exists(model_path):
#     ###################### CREAREA MODELULUI ########################
#     model = FaceClassifier()
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     batch_size = 64  # Adjust as needed
#     epochs = 30  # Adjust as needed

#     # Define callbacks
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
#     model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_face_classifier_model.h5', save_best_only=True,
#                                                           monitor='val_loss')

#     val_pos = prepare_show_data([validation_photos], [val_annotations])
#     image_face_coords = {}
#     for index, row in val_annotations.iterrows():
#         image_name = row['image_name']
#         face_coords = (row['x_min'], row['y_min'], row['x_max'], row['y_max'])

#         if image_name not in image_face_coords:
#             image_face_coords[image_name] = [face_coords]
#         else:
#             image_face_coords[image_name].append(face_coords)
#     negative_patches = []
#     for index, row in val_annotations.iterrows():
#         nr_image = int(row['image_name'].replace('.jpg', ''))
#         current_photo = photos[i][nr_image - 1]
#         current_face_coords = image_face_coords[row['image_name']]
#         neg_vals = extract_negative_patches(current_photo, current_face_coords, patch_size=(96, 96), num_patches=1)
#         if neg_vals:
#             negative_patches.extend(neg_vals)
#     val_pos = np.array(val_pos)
#     val_neg = np.array(negative_patches)

#     val_data = np.concatenate([val_pos, val_neg])
#     positive_labels = np.ones(len(val_pos))
#     negative_labels = np.zeros(len(val_neg))
#     val_labels = np.concatenate([positive_labels, negative_labels])
#     val_data = val_data / 255.0

#     indices = np.arange(len(val_data))
#     np.random.shuffle(indices)
#     val_data = val_data[indices]
#     val_labels = val_labels[indices]

#     # Train the model
#     history = model.fit(
#         shuffled_train_data, shuffled_train_labels,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_data=(val_data, val_labels),
#         callbacks=[early_stopping, model_checkpoint]
#     )

#     # Predict on validation data
#     val_predictions = model.predict(val_data)
#     # Evaluate the model
#     loss, accuracy = model.evaluate(val_data, val_labels)

#     # Print the results
#     print(f"Loss on validation set: {loss}")
#     print(f"Accuracy on validation set: {accuracy}")

#     # Plot training history
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss Over Epochs')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Accuracy Over Epochs')
#     plt.legend()

#     plt.savefig('training_history.png')
#     plt.show()

#     model.save_weights(model_path)
# else:
#     print(f"Model file already exists at {model_path}")
#     model = FaceClassifier()
#     dummy_input = np.zeros((1, 96, 96, 3))
#     model(dummy_input)
#     model.load_weights(model_path)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""ANTRENARE MODEL PENTRU FETE

"""

barney = []
betty = []
fred = []
wilma = []
unknown = []
for i in range(4):
    for index, row in annotations[i].iterrows():
        image_name = row['image_name']
        xmin, ymin, xmax, ymax = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        nr_image = int(row['image_name'].replace('.jpg', ''))
        current_photo = photos[i][nr_image - 1]
        face = current_photo[ymin:ymax, xmin:xmax]
        face = cv.resize(face, (96, 96))
        face = np.array(face) / 255.0
        character = row['character']
        if character == "barney":
            barney.append(face)
        elif character == "betty":
            betty.append(face)
        elif character == "fred":
            fred.append(face)
        elif character == "wilma":
            wilma.append(face)
        else:
            unknown.append(face)

barney = np.array(barney) / 255
betty = np.array(betty) / 255
fred = np.array(fred) / 255
wilma = np.array(wilma) / 255
unknown = np.array(unknown) / 255
# all_negative_samples = all_negative_samples[:1000]
# all_negative_samples = np.array(all_negative_samples) / 255
print(f"fred = {fred.shape}")

from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
    rotation_range=20,        # degrees range for random rotations
    width_shift_range=0.2,    # fraction of total width for horizontal shifts
    height_shift_range=0.2,   # fraction of total height for vertical shifts
    shear_range=0.2,          # shear angle in counter-clockwise direction
    zoom_range=0.2,           # range for random zoom
    horizontal_flip=True,     # randomly flip images horizontally
    fill_mode='nearest'       # strategy for filling in newly created pixels
)

# MODEL PENTRU BARNEY
negative_barney = np.concatenate((fred, betty, wilma, unknown), axis=0)
print(f"barney.shape = {barney.shape}")
flip_barney = np.array([np.fliplr(face) for face in barney])
positive_barney = np.concatenate((barney, flip_barney), axis=0)
positive_barney = np.array(positive_barney)
# Convert lists to numpy arrays for model processing
negative_barney = np.array(negative_barney)
positive_barney = np.array(positive_barney)

print(f"negative_barney = {negative_barney.shape}")
print(f"positive_barney = {positive_barney.shape}")

# # Reshape positive_barney to add an extra dimension, as ImageDataGenerator expects 4D tensors
# positive_barney = positive_barney.reshape((positive_barney.shape[0], 96, 96, 3))

# # Generate augmented images
# augmented_images = data_gen.flow(positive_barney, batch_size=1)

# num_augmented_samples = 3000  # for example, generate 5000 augmented samples
# augmented_barney = [augmented_images.next()[0].astype(np.float32) for _ in range(num_augmented_samples)]
# augmented_barney = np.array(augmented_barney)

# positive_barney = np.concatenate((positive_barney, augmented_barney), axis=0)

labels_positive_barney = np.ones(len(positive_barney))
labels_negative_barney = np.zeros(len(negative_barney))
print(labels_positive_barney.shape, labels_negative_barney.shape)

# Combine data and labels correctly
data_barney = np.concatenate((positive_barney, negative_barney), axis=0)
labels_barney = np.concatenate((labels_positive_barney, labels_negative_barney), axis=0)
print(labels_barney.shape)

from sklearn.utils import shuffle

# Now shuffle the data and labels
data_barney, labels_barney = shuffle(data_barney, labels_barney, random_state=0)

print(data_barney.shape, labels_barney.shape)

"""SA TE UITI MAI JOS CA AI DATE DE VALIDARE PENTRU FIECARE HANDRALAU IN PARTE"""

# from sklearn.model_selection import train_test_split
# positive_barney = None
# negative_barney = None
# labels_positive_barney = None
# labels_negative_barney = None

# # x_train_fred, x_val_fred, y_train_fred, y_val_fred = train_test_split(data_fred, labels_fred, test_size=0.2, random_state=0)



# get validation on fred
val_barney = []
val_betty = []
val_fred = []
val_wilma = []
val_unknown = []
for index, row in val_annotations.iterrows():
    nr_image = int(row['image_name'].replace('.jpg', ''))
    current_photo = validation_photos[nr_image - 1]
    current_photo = cv.resize(current_photo, (96, 96))
    character = row['character']
    # print(character)
    if character == "barney":
        val_barney.append(current_photo)
    elif character == "betty":
        val_betty.append(current_photo)
    elif character == "fred":
        val_fred.append(current_photo)
    elif character == "wilma":
        val_wilma.append(current_photo)
    else:
        val_unknown.append(current_photo)

val_barney = np.array(val_barney) / 255
val_betty = np.array(val_betty) / 255
val_fred = np.array(val_fred) / 255
val_wilma = np.array(val_wilma) / 255
val_unknown = np.array(val_unknown) / 255

print(val_barney.shape, val_betty.shape, val_fred.shape, val_wilma.shape, val_unknown.shape)

# COMBIN DATELE CA SA CREEZ SET PENTRU VALIDARE
from sklearn.utils import shuffle


# Combine the data into a single array for validation
non_barney_data_val = np.concatenate((val_fred, val_betty, val_wilma, val_unknown), axis=0)
validation_data_barney = np.concatenate((val_barney, non_barney_data_val), axis=0)

# Create labels for validation (1 for Fred, 0 for Non-Fred)
val_labels_barney_positive = np.ones(len(val_barney))
val_labels_barney_negative = np.zeros(len(non_barney_data_val))
val_labels_barney = np.concatenate((val_labels_barney_positive, val_labels_barney_negative), axis=0)

# Shuffle the validation data and labels together
validation_data_barney, val_labels_barney = shuffle(validation_data_barney, val_labels_barney, random_state=0)

model_barney = FaceClassifier()

# Compile the model
model_barney.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
batch_size = 64  # Adjust as needed
epochs = 30  # Adjust as needed

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Train the model
history = model_barney.fit(
    data_barney, labels_barney,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_data_barney, val_labels_barney),
    callbacks=[early_stopping]
)

# Predict on validation data
val_predictions = model_barney.predict(validation_data_barney)

# Evaluate the model
loss, accuracy = model_barney.evaluate(validation_data_barney, val_labels_barney)

# Print the results
print(f"Loss on validation set: {loss}")
print(f"Accuracy on validation set: {accuracy}")



# GOLESC VARIABILELE
x_train = []
val_data = []
val_labels = []
negative_patches = []
validation_photos, val_annotations = None, None
photos, annotations = None, None

# VALIDARE

