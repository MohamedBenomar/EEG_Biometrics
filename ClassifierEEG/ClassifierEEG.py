import os
import numpy as np
import time
import sys
import matplotlib
import matplotlib.pyplot as plt

import sklearn
import tensorboard
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import EEG_Biometrics.utilsEEG
from EEG_Biometrics.utilsEEG import generate_results_csv
from EEG_Biometrics.utilsEEG import create_directory
from EEG_Biometrics.utilsEEG import read_dataset
from EEG_Biometrics.utilsEEG import transform_mts_to_ucr_format
from EEG_Biometrics.utilsEEG import visualize_filter
from EEG_Biometrics.utilsEEG import viz_for_survey_paper
from EEG_Biometrics.utilsEEG import calculate_metrics
from EEG_Biometrics.utilsEEG import case_by_case_analysis
from EEG_Biometrics.utilsEEG import viz_cam

class ClassifierEEG(object):
               
    def one_hot_encoder(y_train, y_test):
        
        enc     = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test  = enc.transform(y_test.reshape(-1, 1)).toarray()
        
        return y_train, y_test
    
    
    
    def standardizingDataset(X_train):
        
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        
        return X_train
    
    
    
    #FOR AFTER FITTING THE MODEL, AND WANT TO LOAD THE MODEL AND ANALYZE ITS PERFORMANCE
    def fitted_classifier(samples, classifier_name, output_directory):
        
        #GETTING THE DATASET
        x_test = samples
    
        #nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        nb_classes = 9
    
        #TRANSFROM THE LABELS FROM INTEGERS TO ONE HOT VECTORS
        #y_train, y_test = one_hot_encoder(y_train, y_test)
    
        #standardizing the train/test datasets for each channel
        #x_train_preprocessed = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test_preprocessed  = np.zeros((x_test.shape[0], x_test.shape[1]))
        channels             = [i for i in range(x_test.shape[1])]
        
        for channel in channels:
            x_test_channelData        = x_test[:,channel]
            x_test_channelData        = np.expand_dims(standardizingDataset(x_test_channelData), axis=1)
            x_test_preprocessed[:,channel:channel+1]   = x_test_channelData
            
        x_test  = x_test_preprocessed
    
        #INITIALIZING THE CLASSIFIER TO BE TRAINED ON
        #input_shape = x_train.shape[1:]
        input_shape = (14, 1000)
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    
        #LOADING THE MODEL AND EVALUATE IT USING THE TEST DATASET
        y_pred  = predict(x_test, output_directory, return_df_metrics=False)
    
        return x_test, y_pred
    
    
    
    def predict(x_test, output_directory, return_df_metrics=True):
    
        #LOADING THE MODEL
        start_time = time.time()
        model_path = output_directory + 'best_model.h5'
        model      = keras.models.load_model(model_path)
    
        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=64)
        time2  = time.time()
        print("time to go through the test set:", time2-time1)
        
        if return_df_metrics:
            y_pred     = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            
            return y_pred, df_metrics
        
        return y_pred
    
    
    
    def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
        
      # changed from "from classifier import fcn"    to    "import fcn"
        #GETTING THE CLASSIFIER THAT WE WANT TO BE TRAINED ON THE DATA
        if classifier_name == 'resnet':
            from EEG_Biometrics.ClassifiersModelsEEG import resnet
            return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
        
        if classifier_name == 'inception':
            from EEG_Biometrics.ClassifiersModelsEEG import inception
            return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
        
        if classifier_name == 'eegnet':
            from EEG_Biometrics.ClassifiersModelsEEG import EEGModels
            return EEGModels.Classifier_EEGNET(output_directory, input_shape, nb_classes, verbose)
    
    
    
    def read_dataset(root_dir, dataset_name):
        
        #LOAD THE DATASET AND SPLIT IT INTO TRAIN/TEST DATASETS
        datasets_dict = {}
        cur_root_dir  = root_dir
        file_name     = cur_root_dir + '/' + dataset_name + '/'
        x_train       = np.load(file_name + 'X_train.npy')
        y_train       = np.load(file_name + 'y_train.npy')
        x_test        = np.load(file_name + 'X_test.npy')
        y_test        = np.load(file_name + 'y_test.npy')
    
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
        return datasets_dict
    


'''
if __name__ == "__main__":
    
    #--STARTING CODE ---- 
    root_dir = '/content/gdrive/MyDrive/DeepLearningModels_On_EEG_Data/'
    
    #DIRECTIONS ON THE DATASET AND CLASSIFIER TO BE USED
    pathway = []
    pathway.extend(['BED','eegnet'])  #it can be inception, resnet, eegne
    
    
    #INITIALIZING VARIABLES FOR CREATING A DIRECTORY TO PUT THE RESULTS OF THE MODEL IN
    dataset_name     = pathway[0]
    classifier_name  = pathway[1]
    output_directory = root_dir + '/results/' + classifier_name + '/' +  dataset_name + '/'
    test_dir_df_metrics = output_directory + 'df_metrics.csv'
    
    
    #IF THE MODEL HAS ALREADY BEEN TRAINED ON, THEN WE CAN LOAD IT AND ANALYSIS ITS PERFORMANCE. IF NOT TRAINED YET, THEN WE CAN TRAIN THE MODEL.
    datasets_dict = read_dataset(root_dir, dataset_name)
    x_test, y_true, x_train, y_train, y_test, y_pred, df_metrics = fitted_classifier()
    predictions = case_by_case_analysis(y_true, y_pred) #predictions[0] to see the correct and incorrect predictions.

'''

