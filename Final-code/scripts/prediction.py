"""
CAP5778: Advanced Data Mining
Term-Project
Gyanendra and Soumya
"""
# import the necessary packages
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse
import os
# from utils import get_f1
# text to speech
from gtts import gTTS


IMG_SIZE = 224
# threshold value for cat-dog classifier based on experiemnt
threshold_value = 0.8

# check if the predicted probability value exceeds threshlod for cat-dog classifier
def is_value_gt_threshold(prediction):
    if any(num > threshold_value for num in prediction):
        return True
    else:
        return False

# convert a prediction to labels
def prediction_to_labels(prediction, results):
    # round probabilities to {0, 1}
    # values = prediction.round()
    # labels with the largest probability
    labels = results[np.array(prediction).argmax()]
    return labels 

def scene_prediction(args, image):
    # load the trained model for scene classification
    print("[INFO] Loading scene classification model...")
    model = load_model(args.scene_model, compile=False)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
        loss = 'binary_crossentropy', 
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    results = {
        0:'indoor',
        1:'outdoor'
    }
    print("[INFO] Analyzing scene in user Profile image...")
    pred = model.predict(image)[0]
    # for indoor-outdoor to take max of two predicted probability for indoor-outdoor
    # this technique was taken as feedback from Professor Dr. Shayok Chakraborty
    predicted_scene_label = prediction_to_labels(pred, results)
    print("[INFO] Predicted_scene_label: ", predicted_scene_label)
    return predicted_scene_label

def pet_animal_prediction(args, image):
    # load the trained model for scene classification
    print("[INFO] Loading pet animal  classification model...")
    model = load_model(args.pet_model, compile=False)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = 'binary_crossentropy', 
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    print("[INFO] Analyzing pet animals in user Profile image...")
    results = {
        0:'cat',
        1:'dog'
    }
    pred = model.predict(image)[0]
    # get the normal predicted label
    predicted_to_label = prediction_to_labels(pred, results)
    # apply some threshold to prevent the model from outputting a cat or dog prediction
    # when the input image does not contain either animal
    # the input image has some features that the model has learned to associate with cats or dogs
    # e.g., the background may resemble a typical environment where cats or dogs are found
    # predicted_pet_label = predicted_to_label if is_value_gt_threshold(pred) else ''
    # print("[INFO] Predicted_pet_label: ", predicted_pet_label)
    if is_value_gt_threshold(pred):
        predicted_pet_label = predicted_to_label
        print("[INFO] Predicted_pet_label: ", predicted_pet_label)
    else:
        predicted_pet_label = ''
    return predicted_pet_label

def main(args):
    # load the image
    image = load_img(args.image, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = image / 255
    scene_type = scene_prediction(args, image)
    pet_animal_type = pet_animal_prediction(args, image)
    pet_animal_presence = ''
    scene_info = ''

    if scene_type:
        scene_info = 'Given profile picture was taken in {0}'.format(str(scene_type))

    if pet_animal_type:
        pet_animal_presence = 'The image contains' + str(pet_animal_type)
    
    tts = gTTS(scene_info + '.' + pet_animal_presence)
    head, tail = os.path.split(args.image)
    audio_filename = tail.split('.')[0]
    tts.save(head + '/' + audio_filename + '.mp3')

if __name__ == "__main__":
    # usage: python prediction.py --scene_model model_path/model_name.h5 --pet_model model_path/model_name.h5 --image path_to_img/img
    parser = argparse.ArgumentParser()  
    parser.add_argument('-sm', '--scene_model', help='path to trained indoor outdoor scene classification model', required=True)   
    parser.add_argument('-am', '--pet_model', help='path to trained pet animals calssification model', required=True)            
    parser.add_argument('-i', '--image', help='path to input image', required=True)    
    args = parser.parse_args()
    main(args)
 