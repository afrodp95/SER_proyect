import numpy as np 
import pandas as pd 
import os 
import librosa 
import joblib
#import wave # read and write WAV files
import matplotlib.pyplot as plt 

import tensorflow as tf

def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs


ravdess_data = []
ravdess_numeric_labels = []

i = 0
for dirname, _, filenames in os.walk('./RAVDESS/'):
    if i>0:
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            print("Convirtiendo datos de {}".format(os.path.join(dirname, filename)))
            ravdess_numeric_labels.append(int(filename[7:8]) - 1) # the index 7 and 8 of the file name represent the emotion label
            wav_file_name = os.path.join(dirname, filename)
            ravdess_data.append(extract_mfcc(wav_file_name)) # extract MFCC features/file
    i+=1
        
print("Carga y conversión finalizada")

ravdess_data = np.asarray(ravdess_data)
ravdess_numeric_labels = np.array(ravdess_numeric_labels)
