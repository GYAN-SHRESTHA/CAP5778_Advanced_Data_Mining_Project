"""
CAP5778: Advanced Data Mining
Term-Project
Gyanendra and Soumya
"""
import os
import argparse
import pandas as pd
import shutil
import matplotlib.pyplot as plt

# Function to rename all the images in sub-folder(class in our task)
# with sub-folder name and give indexing
# e.g., for cat rename images as cat_0.jpg cat_1.jpg
def rename_image_filename(input_dir):
    # list of all subdirectories in the root folder
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    # print the list of subdirectories
    print("[INFO] List of subdirectories: \n", subdirs)

    for subdir in subdirs:
        # get the full path of the subfolder
        subfolder_path = os.path.join(input_dir, subdir)
        # loop through subdirectories
        for sub_path, directories, filenames in os.walk(subfolder_path):
            # loop through all files in the subdirectory
            for index, filename in enumerate(filenames):
                # get the full path of the file
                filepath = os.path.join(subfolder_path, filename)
                # get the subdirectory name
                subfolder_name = os.path.basename(subfolder_path)
                # create the new filename with the subdirectory name and index
                new_filename = subfolder_name + "." + str(index) + ".jpg"
                # rename the file with the new filename
                os.rename(filepath, os.path.join(sub_path, new_filename))
    return subdirs

# Function to merge all the images into one folder
def merge_files(args):
    # input_dir(args.output) is path to the source folder
    dest_folder_path = args.output + '/' + args.train
    # create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
    # loop through all subdirectories in the input folder
    for path, dirs, files in os.walk(input_dir):
        # loop through all files in the subdirectory
        for filename in files:
            # get the full path of the file
            filepath = os.path.join(path, filename)
            # create the new filepath in the destination folder
            new_filepath = os.path.join(dest_folder_path, filename)
            # copy the file to the destination folder
            shutil.copy2(filepath, new_filepath)
    return dest_folder_path

# usage: python data_preprocessing.py --dataset input_directory_path_containg_images_in_sub_folders_for_different_clasess --output output_directory_path --csvfile csv_filename --train folder_name_to_store_labelled_images
# python data_preprocessing.py --dataset /Users/gyanendra/Spring2023/data_mining/Term_Project/pet_animals --output /Users/gyanendra/Spring2023/data_mining/Term_Project/dog-cat-train --csvfile  dog_cat_labels.csv --train train
parser = argparse.ArgumentParser()  
parser.add_argument('-d', '--dataset', help='path to input dataset (i.e., directory of images)', required=True)      
parser.add_argument('-o', '--output', help='path to output dataset (i.e., directory of images and csv file)', required=True)  
parser.add_argument('-c', '--csvfile', help='output csv labelling file name', required=True)     
parser.add_argument('-f', '--train', help='folder name where all training images are stored', required=True)     
args = parser.parse_args()

input_dir = args.dataset
output_dir = args.output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("[INFO] Renaming images in each sub-folder...")
# rename_image_filename(input_dir)
print("[INFO] Merge all input images into one folder...")
dest_folder_path = merge_files(args)

# dest_folder_path = args.output + '/' + args.train
filenames = os.listdir(dest_folder_path)
filename_with_label = []
for filename in filenames:
    label = filename.split('.')[0]
    if label == 'dog':
        filename_with_label.append([filename, 'dog'])
    elif label == 'cat':
        filename_with_label.append([filename, 'cat'])
    elif label == 'indoor':
        filename_with_label.append([filename, 'indoor'])
    elif label == 'outdoor':
        filename_with_label.append([filename, 'outdoor'])
print("[INFO] Total input images: ", len(filename_with_label))
# create csv file for labelled datasets
pd.DataFrame(filename_with_label, columns=['file_name', 'label']).to_csv(output_dir + '/' + args.csvfile)
csv_path = os.path.join(output_dir + '/'+ args.csvfile)
df = pd.read_csv(csv_path)
print("[INFO] First 10 rows in labelled datasets(dataframe): \n", df.head(10))
print("[INFO] labelled datasets(dataframe):", df.shape)