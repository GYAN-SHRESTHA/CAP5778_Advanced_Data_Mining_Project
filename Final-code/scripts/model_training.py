"""
Some part of training code is taken from https://pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
https://medium.com/wwcode-python/is-it-a-cat-or-a-dog-python-machine-learning-tutorial-for-image-recognition-1fa76149265c

CAP5778: Advanced Data Mining
Term-Project
Gyanendra and Soumya
"""

import os
import numpy as np
import pandas as pd
import argparse
import random
import time
# For building deep learning models
import tensorflow as tf
import keras  
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# Pre-trained models
from tensorflow.keras.applications import VGG16  # For using pre-trained models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator # For image data augmentation
from keras.models import Model
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from utils import get_f1

IMG_SIZE = 224
BATCH_SIZE = 32

# Function to separate datasets into training and testing sets
def train_test_dataset(data, labels):
    # separate into train and test datasets uisng 80% for training and 20% for testing
    print("[INFO] Splitting images into train-test sets...")
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, testX, trainY, testY

# Defining CNN model based on VGG16
def define_model(labels):
    print("[INFO] Initializing model...")
    # include_top is to include the classifier layer; whether to include the 3 fully-connected layers at the top of the network
    # training with Imagenet weights
    # removing the last layer that is classifying 1000 images this will be replaced with images classes we have
    base_model = VGG16(
        include_top = False, 
        weights = 'imagenet',
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        classes=len(labels)
    )
    # mark layers as not trainable except last four we train the last four layers with our data
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    last_output = base_model.output
    x = GlobalAveragePooling2D()(last_output)
    x = Dense(512, activation = 'relu')(x)
    outputs = Dense(units=len(labels), activation = 'sigmoid')(x)
    model = Model(inputs = base_model.inputs, outputs = outputs)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
        loss = 'binary_crossentropy', 
        metrics=['accuracy', 'AUC', 'Precision', 'Recall', get_f1]
    )
    return model

# create a one hot encoding for labels
def one_hot_encode(labels, mapping):
    # print("[INFO] One hot encoding...")
    # create empty vector
    encoding = np.zeros(len(mapping), dtype='uint8')
    # mark 1 for each label in the vector
    encoding[mapping[labels]] = 1
    return encoding

# Loading all images
def load_images(args, file_mapping, label_mapping):
    print("[INFO] Loading images...")
    data = list()
    labels = list()
    folder_path = args.dataset + '/' + args.train + '/'
    # enumerate files in the directory
    for filename in os.listdir(folder_path):
        if not filename.startswith('.') and os.path.isfile(os.path.join(folder_path, filename)):
            image = load_img(folder_path + filename, target_size=(IMG_SIZE, IMG_SIZE))
            # convert to numpy array
            image = img_to_array(image, dtype='uint8')
            # get target class of each filename
            target_class = file_mapping[filename]
            # one hot encode target_class
            label = one_hot_encode(target_class, label_mapping)
            data.append(image)
            labels.append(label)
        else:
            print("[WARN] wrong path..", folder_path + filename)
    X = np.asarray(data, dtype='uint8')
    y = np.asarray(labels, dtype='uint8')
    return X, y

def main(args):
    print("[INFO] Reading labelled csv file...")
    csv_path = os.path.join(args.dataset + '/' + args.csvfile)
    df = pd.read_csv(csv_path)
    print("[INFO] First 10 rows in labelled datasets(dataframe): \n", df.head(10))
    print("[INFO] labelled datasets(dataframe):", df.shape)

    labels = list()
    for index, row in df.iterrows():
        labels.append(row['label'])
    
    sorted_labels = []
    [sorted_labels.append(x) for x in labels if x not in sorted_labels]
    sorted_labels.sort()
    # print(sorted_labels)
    # mapping lables to interger i.e., label mapping
    labels_mapping = {sorted_labels[i]:i for i in range(len(sorted_labels))}
    # print("labels_map: ", labels_mapping)'
    file_mapping = dict()
    for i in range(len(df)):
        name, tags = df['file_name'][i], df['label'][i]
        file_mapping[name] = tags
    # print("file_mapping: ", file_mapping)
    X, Y = load_images(args, file_mapping, labels_mapping)
    print(X.shape, Y.shape)
    # dataset splitting 
    trainX, testX, trainY, testY = train_test_dataset(X, Y)
    # construct the image generator for data augmentation
    # ref https://towardsdatascience.com/how-to-augmentate-data-using-keras-38d84bd1c80c
    train_gen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=25, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True, 
        fill_mode="nearest")
    test_gen = ImageDataGenerator(rescale=1.0/255.0)

    # create iterators from image generator for train sets
    train_generator = train_gen.flow(
        trainX,
        trainY,
        batch_size=BATCH_SIZE,
        shuffle=True, seed=42
    )
    # create iterators from image generator for test sets
    validation_generator = test_gen.flow(
        testX, 
        testY, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        seed=42
    )
    print("STEP_SIZE_TRAIN: ",len(train_generator))
    print("STEP_SIZE_VALID: ", len(validation_generator))

    # define model
    model = define_model(sorted_labels)

    print("[INFO] Training Started...")
    model_name = args.model
    checkpoint = ModelCheckpoint(
        model_name, 
        monitor = 'val_loss', 
        mode = 'min', 
        save_best_only = True, 
        verbose = 1
    )
    # early stop to prevent overfitting
    earlystopping = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0, 
        patience = 5, 
        verbose = 1, 
        restore_best_weights = True
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=20,
        callbacks = [checkpoint, earlystopping]
    )
    
    model.save(model_name)
    print("[INFO] Training Completed...")

if __name__ == "__main__":
    
    # usage: python model_training.py --dataset dataset_directory of csv file and traiing images --model model_directory/model_name.h5 --csvfile  labelled_csv_filename --train folder_name
    parser = argparse.ArgumentParser()  
    parser.add_argument('-d', '--dataset', help='path to input dataset (i.e., root directory of image folder and csv file)', required=True)    
    parser.add_argument('-m', '--model', help='path to output model', required=True) 
    parser.add_argument('-c', '--csvfile', help='csv labelling file name', required=True)     
    parser.add_argument('-f', '--train', help='folder name where all training images are stored', required=True) 
    args = parser.parse_args()
    # get the start time
    start = time.time()
    main(args)
    end = time.time()
    # get the execution time
    elapsed_time = end - start
    print('Execution time:', elapsed_time, 'seconds')
