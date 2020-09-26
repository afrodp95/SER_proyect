import numpy as np 
import pandas as pd 
import os 
import librosa 
import joblib
#import wave # read and write WAV files
import matplotlib.pyplot as plt 


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
#from tensorflow.keras.optimizers import rmsprop
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

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

x_train.shape
x_test.shape
y_train.shape


### Lstm
model = Sequential()
model.add(layers.LSTM(256,input_shape=(40,1)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(8,activation='softmax'))
model.add(layers.Dropout(0.2))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
train_hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=200,shuffle=True)




def create_model():
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    
    # Configures the model for training
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model