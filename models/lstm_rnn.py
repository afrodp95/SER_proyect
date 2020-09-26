import numpy as np 
import pandas as pd 
import os 
import librosa 
import joblib
#import wave # read and write WAV files
import matplotlib.pyplot as plt 


import tensorflow as tf
from sklearn.model_selection import train_test_split

### Load data
ravdess_data = joblib.load('./models/ravdess_speech_data.gz')
ravdess_target = joblib.load('./models/ravdess_target.gz')
ravdess_numeric_labels = joblib.load('./models/ravdess_target.gz')

### Train test split
x_train,x_test,y_train,y_test= train_test_split(np.array(ravdess_data),
                                                ravdess_target,
                                                stratify=ravdess_numeric_labels,
                                                test_size=0.20, random_state=123)

### Lstm
