import os.path
from time import sleep
from PIL import Image
import cv2 as cv
import numpy as np
import glob
import pandas as pd
from skimage.feature import hog
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras


def extract_hog_features(image_patch):
    features = hog(image_patch, pixels_per_cell=(6, 6), cells_per_block=(2, 2))
    return features

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


    print("cate fete sunt")
    print(len(x_train))
    return x_train

def read_data():
    train_path = os.path.join("CAVA-2023-TEMA2", "antrenare")
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
    current_path = os.path.join(path, "*.jpg")
    jpg_files = sorted(glob.glob(current_path))  # Sort the file paths
    validation_photos = []
    for file_path in jpg_files:
        image = cv.imread(file_path)
        if image is not None:
            validation_photos.append(image)
        else:
            print(f"Error reading image at path: {file_path}")
    path = os.path.join("CAVA-2023-TEMA2", "validare", "task1_gt_validare.txt")
    df = read_bounding_boxes(path)
    return validation_photos, df

def read_test_data():
    path = os.path.join("CAVA-2023-TEMA2", "fake_test-2")
    current_path = os.path.join(path, "*.jpg")
    jpg_files = sorted(glob.glob(current_path))  # Sort the file paths
    test_photos = []
    for file_path in jpg_files:
        image = cv.imread(file_path)
        if image is not None:
            test_photos.append(image)
        else:
            print(f"Error reading image at path: {file_path}")
    return test_photos
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
# 2700508

def get_outsider_negatives(num_photos):
    current_path = os.path.join("exempleNegative", "*.jpg")
    jpg_files = sorted(glob.glob(current_path))  # Sort the file paths
    imgs = []
    for i in range(num_photos):
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

class FaceClassifier(tf.keras.Model):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3))
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



if __name__ == '__main__':
    if not os.path.exists("partial_solution"):
        os.makedirs("partial_solution")
        print("Directory 'partial_solution' created.")
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
    # Collect face coordinates for each image
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
            # Get all face coordinates for the current image
            current_face_coords = image_face_coords[row['image_name']]
            # Extract a negative patch that doesn't overlap with any faces
            if len(negative_patches) < 18000:
                neg_vals = extract_negative_patches(current_photo, current_face_coords, patch_size=(96, 96), num_patches=5)
            else:
                neg_vals = extract_negative_patches(current_photo, current_face_coords, patch_size=(96, 96),num_patches=4)
            if neg_vals:  # Check if a negative patch was successfully extracted
                # show_image("Negative patch", neg_vals[0])
                negative_patches.extend(neg_vals)

    negative2_path = "train/"
    train_negative = []
    for elem in os.listdir(negative2_path):
        file_path = os.path.join(negative2_path, elem)
        image = cv.imread(file_path)
        # Verifică dacă imaginea a fost încărcată corect
        if image is not None:
            # Redimensionează imaginea
            resized_image = cv.resize(image, (96, 96))
            train_negative.append(resized_image)

    #negative_patches = negative_patches[:15000]
    # get negative
    negative_images = get_outsider_negatives(num_photos=274)
    # Create labels for positive and negative samples
    positive_labels = np.ones(len(x_train))
    negative_labels = np.zeros(len(negative_patches) + len(negative_images) + len(train_negative))

    # Combine negative samples
    all_negative_samples = negative_patches + negative_images + train_negative
    # Combine all data and labels
    x_train = np.array(x_train)

    flipped_data = [np.fliplr(face) for face in x_train]
    x_train = x_train + flipped_data

    x_train = np.array(x_train)
    print(f"x_train.shape = {x_train.shape}")

    all_negative_samples = np.array(all_negative_samples)
    print(f"all_negative_samples.shape = {all_negative_samples.shape}")
    all_data = np.concatenate([x_train, all_negative_samples], axis=0)
    all_labels = np.concatenate([positive_labels, negative_labels], axis=0)
    # Convert to NumPy arrays (if not already) and normalize
    all_data = np.array(all_data) / 255.0
    all_labels = np.array(all_labels)
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    # Step 3: Use the shuffled indices to reorder the data and labels
    shuffled_train_data = all_data[indices]
    shuffled_train_labels = all_labels[indices]


    # CITIRE DATE VALIDARE
    validation_photos, val_annotations = read_validation_data()

    ####################### VERIFICARE DACA ESTE SALVAT DEJA MODELUL ######
    model_path = 'cnn_saved_model.h5'
    if not os.path.exists(model_path):
        ###################### CREAREA MODELULUI ########################
        model = FaceClassifier()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        batch_size = 64  # Adjust as needed
        epochs = 30  # Adjust as needed

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_face_classifier_model.h5', save_best_only=True,
                                                              monitor='val_loss')

        val_pos = prepare_show_data([validation_photos], [val_annotations])
        image_face_coords = {}
        for index, row in val_annotations.iterrows():
            image_name = row['image_name']
            face_coords = (row['x_min'], row['y_min'], row['x_max'], row['y_max'])

            if image_name not in image_face_coords:
                image_face_coords[image_name] = [face_coords]
            else:
                image_face_coords[image_name].append(face_coords)
        negative_patches = []
        for index, row in val_annotations.iterrows():
            nr_image = int(row['image_name'].replace('.jpg', ''))
            current_photo = photos[i][nr_image - 1]
            # Get all face coordinates for the current image
            current_face_coords = image_face_coords[row['image_name']]
            # Extract a negative patch that doesn't overlap with any faces
            neg_vals = extract_negative_patches(current_photo, current_face_coords, patch_size=(96, 96), num_patches=1)
            if neg_vals:  # Check if a negative patch was successfully extracted
                # show_image("Negative patch", neg_vals[0])
                negative_patches.extend(neg_vals)
        val_pos = np.array(val_pos)
        val_neg = np.array(negative_patches)

        val_data = np.concatenate([val_pos, val_neg])

        # Step 2: Create labels (1 for positive, 0 for negative)
        positive_labels = np.ones(len(val_pos))
        negative_labels = np.zeros(len(val_neg))
        val_labels = np.concatenate([positive_labels, negative_labels])

        # Step 3: Normalize the data
        val_data = val_data / 255.0

        # Step 4: Shuffle the data and labels in unison
        # Create an array of indices and shuffle them
        indices = np.arange(len(val_data))
        np.random.shuffle(indices)

        # Use the shuffled indices to shuffle the data and labels
        val_data = val_data[indices]
        val_labels = val_labels[indices]

        # Train the model
        history = model.fit(
            shuffled_train_data, shuffled_train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            callbacks=[early_stopping]
        )

        # Predict on validation data
        val_predictions = model.predict(val_data)
        # Evaluate the model
        loss, accuracy = model.evaluate(val_data, val_labels)
        # for i in range(len(val_labels)):
        #     loss, accuracy = model.evaluate(np.array([val_data[i]]), np.array([val_labels[i]]), verbose=0)
        #     print(f"Accuracy on validation set: {accuracy}")
        #     if accuracy == 0.0:
        #         print(i)

        # Print the results
        print(f"Loss on validation set: {loss}")
        print(f"Accuracy on validation set: {accuracy}")

        # Save the final model
        # model.save(model_path, save_format='tf')
        # model.save("test", save_format='h5')

        # Optional: Plot training history

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.savefig('training_history.png')
        # Display the plot
        plt.show()
        # plt.show()
        model.save_weights(model_path)
        # print(f"Model saved to {model_path}")
    else:
        print(f"Model file already exists at {model_path}")
        model = FaceClassifier()
        # Create a dummy input - the shape should match the input shape of your model
        dummy_input = np.zeros((1, 96, 96, 3))  # Batch size of 1, and input shape (96, 96, 3)
        # Run a forward pass with the dummy input to build the model
        model(dummy_input)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Parameters for variable sliding window
    minWindowSize = (40, 40)  # Example minimum window size
    maxWindowSize = (130, 130)  # Example maximum window size
    stepSize = 20  # Example step size, adjust to your needs
    scale = 1.2  # Example scale factor for window size increase

    train_folder = 'train'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    # Process each image with variable sliding window


    # GOLESC VARIABILELE
    x_train = []
    val_data = []
    val_labels = []
    negative_patches = []
    #
    # LOAD FRED MODEL
    fred_model = FaceClassifier()
    # Create a dummy input - the shape should match the input shape of your model
    dummy_input = np.zeros((1, 96, 96, 3))  # Batch size of 1, and input shape (96, 96, 3)
    # Run a forward pass with the dummy input to build the model
    fred_model(dummy_input)
    fred_model.load_weights('fred_face_classifier_weights3.h5')
    fred_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # BARNEY
    barney_model = FaceClassifier()
    # Create a dummy input - the shape should match the input shape of your model
    dummy_input = np.zeros((1, 96, 96, 3))  # Batch size of 1, and input shape (96, 96, 3)
    # Run a forward pass with the dummy input to build the model
    barney_model(dummy_input)
    barney_model.load_weights('barney_face_classifier_weights.h5')
    barney_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # WILMA
    wilma_model = FaceClassifier()
    # Create a dummy input - the shape should match the input shape of your model
    dummy_input = np.zeros((1, 96, 96, 3))  # Batch size of 1, and input shape (96, 96, 3)
    # Run a forward pass with the dummy input to build the model
    wilma_model(dummy_input)
    wilma_model.load_weights('wilma_face_classifier_weights2.h5')
    wilma_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # BETTY
    betty_model = FaceClassifier()
    # Create a dummy input - the shape should match the input shape of your model
    dummy_input = np.zeros((1, 96, 96, 3))  # Batch size of 1, and input shape (96, 96, 3)
    # Run a forward pass with the dummy input to build the model
    betty_model(dummy_input)
    betty_model.load_weights('betty_face_classifier_weights2.h5')
    betty_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    detections_fred = []
    scores_fred = []
    file_names_fred = []

    detections_barney = []
    scores_barney = []
    file_names_barney = []

    detections_wilma = []
    scores_wilma = []
    file_names_wilma = []

    detections_betty = []
    scores_betty = []
    file_names_betty = []

    # VALIDARE
    nr = 1
    lungime_totala = 4
    detections = []
    scores = []
    file_names = []
    test_photos = read_test_data()
    # for elem in test_photos:
    #     show_image("test_image", elem)
    for index, photo in enumerate(test_photos, start=0):
        nume_fisier = str(nr).zfill(lungime_totala) + ".jpg"
        print(nume_fisier)
        print(f"index = {index}")
        nr += 1
        prediction_list = []
        image_detection = []
        photos_detection_array = []
        # show_image("Initial image", photo)
        for (x, y, window) in variable_sliding_window(photo, minWindowSize, maxWindowSize, stepSize, scale):
            if window.shape[0] < minWindowSize[1] or window.shape[1] < minWindowSize[0]:
                continue

            resized_window = cv.resize(window, (96, 96))
            normalized_window = resized_window / 255.0
            normalized_window = np.expand_dims(normalized_window, axis=0)

            # Clasifică fereastra
            prediction = model.predict(normalized_window)[0][0]
            # Check if the prediction is above the threshold
            if prediction > 0.80 :
                # Crop the face from the original photo
                # face = photo[y:y + window.shape[0], x:x + window.shape[1]]
                xmin, ymin, xmax, ymax = x, y, x + window.shape[1], y + window.shape[0]
                # Crop the face from the original photo
                face = photo[ymin:ymax, xmin:xmax]
                if is_skin_color_present(face):

                    prediction_list.append(prediction)
                    image_detection.append([xmin, ymin, xmax, ymax])

        if len(image_detection) > 1:
            max_detection, max_score = non_maximal_suppression(np.array(image_detection), np.array(prediction_list),
                                                           photo.shape[:2])
        else:
            print("fara non-max sup")
            max_detection = np.array(image_detection)
            max_score = np.array(prediction_list)

        print(f"max detection = {max_detection}")
        print(f"max score = {max_score}")
        nr_val = 0
        for idx, elem in enumerate(max_detection):
            xmin, ymin, xmax, ymax = elem
            face = photo[ymin:ymax, xmin:xmax]
            nr_val += 1
            print(max_score[idx])
            show_image("non_maximal_suppression " + str(index), face)
            detections.append(np.array(elem))
            scores.append(max_score[idx])
            file_names.append(nume_fisier)

            #normalizare a intrarii
            face = cv.resize(face, (96, 96))
            face_resize = face.copy()
            # normalize it
            face = np.array(face) / 255
            face_batch = np.expand_dims(face, axis=0)

            # pentru personaje
            prediction_wilma = wilma_model.predict(face_batch)[0][0]
            prediction_fred = fred_model.predict(face_batch)[0][0]
            prediction_barney = barney_model.predict(face_batch)[0][0]
            prediction_betty = betty_model.predict(face_batch)[0][0]

            # identific predictia maxima
            maxim_prediction = max(prediction_wilma, prediction_betty, prediction_barney, prediction_fred)
            # maxim_prediction = round(maxim_prediction, 3)

            if maxim_prediction == prediction_barney or prediction_fred - prediction_barney < 0.006:
                detections_barney.append(elem)
                scores_barney.append(maxim_prediction)
                file_names_barney.append(nume_fisier)
                print("BARNEY")
            elif maxim_prediction == prediction_fred:
                detections_fred.append(elem)
                scores_fred.append(maxim_prediction)
                file_names_fred.append(nume_fisier)
                print("FRED")
            elif maxim_prediction == prediction_wilma:
                detections_wilma.append(elem)
                scores_wilma.append(maxim_prediction)
                file_names_wilma.append(nume_fisier)
                print("WILMA")
            elif maxim_prediction == prediction_betty:
                detections_betty.append(elem)
                scores_betty.append(maxim_prediction)
                file_names_betty.append(nume_fisier)
                print("BETTY")

        if nr % 10 == 0:
            detections_part = np.array(detections)
            scores_part = np.array(scores)
            file_names_part = np.array(file_names)
            np.save("partial_solution/detections_all_faces.npy", detections_part)
            np.save("partial_solution/scores_all_faces.npy", scores_part)
            np.save("partial_solution/file_names_all_faces.npy", file_names_part)

    # Convert lists to numpy arrays
    detections = np.array(detections)
    scores = np.array(scores)
    file_names = np.array(file_names)

    task1_path = "task1"

    if not os.path.exists(task1_path):
        os.mkdir(task1_path)
    # Save the results in the expected format
    np.save("task1/detections_all_faces.npy", detections)
    np.save("task1/scores_all_faces.npy", scores)
    np.save("task1/file_names_all_faces.npy", file_names)

    ##### TASK 2
    detections_fred = np.array(detections_fred)
    scores_fred = np.array(scores_fred)
    file_names_fred = np.array(file_names_fred)

    detections_barney = np.array(detections_barney)
    scores_barney = np.array(scores_barney)
    file_names_barney = np.array(file_names_barney)

    detections_wilma = np.array(detections_wilma)
    scores_wilma = np.array(scores_wilma)
    file_names_wilma = np.array(file_names_wilma)

    detections_betty = np.array(detections_betty)
    scores_betty = np.array(scores_betty)
    file_names_betty = np.array(file_names_betty)

    task2_path = "task2"

    if not os.path.exists(task2_path):
        os.mkdir(task2_path)
    np.save("task2/detections_fred.npy", detections_fred)
    np.save("task2/scores_fred.npy", scores_fred)
    np.save("task2/file_names_fred.npy", file_names_fred)

    # Save Barney's data
    np.save("task2/detections_barney.npy", detections_barney)
    np.save("task2/scores_barney.npy", scores_barney)
    np.save("task2/file_names_barney.npy", file_names_barney)

    # Save Wilma's data
    np.save("task2/detections_wilma.npy", detections_wilma)
    np.save("task2/scores_wilma.npy", scores_wilma)
    np.save("task2/file_names_wilma.npy", file_names_wilma)

    # Save Betty's data
    np.save("task2/detections_betty.npy", detections_betty)
    np.save("task2/scores_betty.npy", scores_betty)
    np.save("task2/file_names_betty.npy", file_names_betty)






