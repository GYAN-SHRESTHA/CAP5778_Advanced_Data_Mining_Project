Folder/File structure
Scripts : folder containing scripts
	data_preprocessing.py : script for data preparation for training 
	model_training.py : script for training classification model
        prediction.py : script to test input image on trained model and output
	indoor_outdoor_model.h5 : our trained model for indoor-outdoor classification
	cat_dog_model.h5 :our trained model for indoor-outdoor classification
	model_training.ipynb : notebook file that we used for training our model in google colab
	utils.py : to calculate F1-score

pet_animals : folder for data collection
	dog : sub-folder for collected images of dog
	cat : sub-folder for collected images of cat

dog-cat-train: : folder containing label csv file and training images
	dog_cat_labels.csv
	train : sub-folder containing images for training
		cat.jpg
		dog.jpg

indoor-outdoor : folder for data collection
	indoor : sub-folder collected images of indoor
	outdoor : sub-folder collected images of outdoor

indoor-outdoor-train : folder containing label csv file and training images
	indoor_outdoor_labels.csv
	train : sub-folder containing images for training
		indoor.jpg
		outdoor.jpg
test-images: final output directory
	test.jpg : input test image
	test.mp3 : output mp3 file

README.txt
requirement.txt : list of required modules/python library


Environment setup:  
    First Install Anaconda or Miniconda normally

    In terminal you can create environment using the command:
	    conda create --name your_environment_name python=3
	    e.g. conda create --name venvt python=3
 
    Once the environment is created activate using 
	    conda activate your_environment_name
	    conda activate venv

    Install keras and tensorflow
	    conda install -c conda-forge keras
	    conda install -c conda-forge tensorflow 

    Install required modules
	    python3 -m pip install -r requirement.txt

Data-preprocessiong
	usage: python data_preprocessing.py --dataset input_directory_path_containg_images_in_sub_folders_for_different_clasess --output output_directory_path --csvfile csv_filename --train folder_name_to_store_labelled_images
	e.g. python data_preprocessing.py --dataset /Users/gyanendra/Spring2023/data_mining/Term_Project/pet_animals --output /Users/gyanendra/Spring2023/data_mining/Term_Project/dog-cat-train --csvfile  dog_cat_labels.csv --train train

Model training 
	usage: python model_training.py --dataset directory_of_labelled_csv_file_and_training images --model model_directory/model_name.h5 --csvfile labelled_csv_filename --train folder_name
	e.g. python model_training.py --dataset  /Users/gyanendra/Spring2023/data_mining/Term_Project/dog-cat-train --model test_model.h5 --csvfile dog_cat_labels.csv --train train

Model testing
	Since trained models are already in scripts folder we can directly use those for testing new images
	The audio output is generated with same name as input test image with ‘.mp3’ extension in the test-images directory

	usage: python prediction.py --scene_model model_path/model_name.h5 --pet_model model_path/model_name.h5 --image path_to_img/img
	e.g. python prediction.py --scene_model indoor_outdoor_model.h5 --pet_model cat_dog_model.h5 --image /Users/gyanendra/Spring2023/data_mining/Term_Project/test-images/Sample_1.jpg

	Output: Sample_1.mp3
