#from IPython import get_ipython
#import os
#get_ipython().magic('reset -f')
#os.system("clear")

import time
import threading

import numpy as np
import pandas as pd

#from EEG_Biometrics.ppeeg import PreProcessingEEG
from EEG_Biometrics.prep import Epoch
from EEG_Biometrics.RaspberryPiADS1299 import ADS1299_API
#from EEG_Biometrics.pyOpenBCI import cyton
from EEG_Biometrics.ClassifiersModelsEEG import EEGModels, inception, resnet
from EEG_Biometrics.ClassifierEEG import ClassifierEEG


DEBUG = False

PROCESS = "TEST"
#PROCESS = "BCI"
#PROCESS = "ADS1299"

if PROCESS == "BCI": board = cyton.OpenBCICyton(port=None)
if PROCESS == "ADS1299": ads = ADS1299_API()

buffer = []
buffer_prep = []
buffer_pred = []

sampling_rate = 250
sample_duration = 4  #1000 samples = 4 sec (1000/250)
test_duration = 8

directory = './Classifiers_Trained_Models/' + classifier_name + '/'
classifier_name  = "eegnet"



def add_buffer(sample):
    global buffer

    if PROCESS == "BCI":
        uVolts_per_count = (4500000)/24/(2**23-1)
        buffer.append([i*uVolts_per_count for i in sample.channels_data])

    elif PROCESS == "TEST":
        buffer.append(sample)



def stream():
    global PROCESS, board, ads

    if PROCESS == "TEST":
        header = "F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4".split()
        test = pd.read_csv("/Users/MohamedBenomar/Desktop/ETSETB/MEE/2B/TFM/RasPi-Files/RAW_REC/s1_s1.csv")
        test = test[header].iloc[0:(sampling_rate*test_duration)].values.tolist()
        for x in test:
            add_buffer(x)
            time.sleep(1/sampling_rate)

    elif PROCESS == "BCI":
        board.start_stream(add_buffer)

    elif PROCESS == "ADS1299":
        ads.openDevice()
        ads.registerClient(add_buffer)
        ads.configure(sampling_rate=sampling_rate)
        ads.startEegStream()



def stop():
    global board
    time.sleep(10)
    board.stop_stream()



def main():
    global buffer, sample_duration, buffer_prep, DEBUG, classifier_name, directory

    s = time.localtime()
    current_time = time.strftime("%H:%M:%S", s)
    print("\n\nStarting Time: {} \n".format(current_time))

    i = 1
    while True:

        samples, buffer = (np.asarray(buffer[0:(sample_duration*sampling_rate)]).transpose(), buffer[(sample_duration*sampling_rate)::]) if len(buffer) >= (sample_duration*sampling_rate) else (np.empty((0,0)), buffer)

        if samples.shape[1] >= (sample_duration*sampling_rate):
            if DEBUG: print("\n\nIteration: {}\n\n".format(i))
            if DEBUG: print("Sample: {}\n".format(samples.shape))
            epoch = Epoch(samples, event=None)
            epoch.fit()
            EEG_cleaned = epoch.numpy_array

            if DEBUG: print(EEG_cleaned)
            buffer_prep.append(EEG_cleaned)

            #x_test, y_pred = ClassifierEEG.fitted_classifier(EEG_cleaned, classifier_name, directory)
            #buffer_pred.append(y_pred)

            i += 1

        if i > (test_duration/sample_duration):
            e = time.localtime()
            d=time.mktime(e)-time.mktime(s)
            current_time = time.strftime("%H:%M:%S", e)
            print("\n\n\nEnding Time: {} \nTotal Time: {} sec ({} min)".format(current_time, d, d/60))
            break



if __name__ == "__main__":
    thread_stream  = threading.Thread(target=stream)
    #thread_stop    = threading.Thread(target=stop)
    thread_main   = threading.Thread(target=main)

    thread_stream.start()
    #thread_stop.start()
    thread_main.start()

    thread_stream.join()
    #thread_stop.join()
    thread_main.join()
    if DEBUG: print("\nSTOP")
