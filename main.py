import time
import threading

import numpy as np
import pandas as pd

from EEG_Biometrics.prep import Epoch
from EEG_Biometrics.RaspberryPiADS1299 import ADS1299_API
from EEG_Biometrics.ClassifiersModelsEEG import EEGModels, inception, resnet
from EEG_Biometrics.ClassifierEEG import ClassifierEEG


ads = ADS1299_API()

buffer_input = []
buffer_prep = []
buffer_output = []

sampling_rate = 250
sample_duration = 4  #1000 samples = 4 sec (1000/250)
streaming_time = 300 # 300sec = 5min

directory = './Classifiers_Trained_Models/' + classifier_name + '/'
classifier_name  = "eegnet" # or "resnet" or "inception"



def add_buffer(sample):
    global buffer_input

    buffer_input.append(sample)
        


def stream():
    global ads, sampling_rate
    
    ads.openDevice()
    ads.registerClient(add_buffer)
    ads.configure(sampling_rate=sampling_rate)
    ads.startEegStream()
         
        

def stop():
    global ads, streaming_time
    
    time.sleep(streaming_time) 
    ads.stopStream()
    ads.closeDevice()
    

    
def main():
    global ads, buffer_input, buffer_prep, buffer_output, sample_duration, classifier_name, directory

    time.sleep(1)
    while ads.stream_active:
        
        ## Get first complete sample
        samples, buffer_input = (np.asarray(buffer_input[0:(sample_duration*sampling_rate)]).transpose(), buffer_input[(sample_duration*sampling_rate)::]) if len(buffer_input) >= (sample_duration*sampling_rate) else (np.empty((0,0)), buffer_input)  
        
        if samples.shape[1] >= (sample_duration*sampling_rate):
            ## PREP Pipeline
            epoch = Epoch(samples, event=None)
            epoch.fit()
            EEG_cleaned = epoch.numpy_array
            buffer_prep.append(EEG_cleaned)
            
            ## EEG Classification
            x_test, y_pred = ClassifierEEG.fitted_classifier(EEG_cleaned, classifier_name, directory)
            buffer_output.append(y_pred)
            
            
    
if __name__ == "__main__":
    thread_stream  = threading.Thread(target=stream)
    thread_stop    = threading.Thread(target=stop)
    thread_main   = threading.Thread(target=main)

    thread_stream.start()
    thread_stop.start()
    thread_main.start()

    thread_stream.join()
    thread_stop.join()
    thread_main.join()