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
from tensorflow.keras.utils import plot_model

### Load data
ravdess_data = joblib.load('./models/ravdess_speech_data.gz')
ravdess_target = joblib.load('./models/ravdess_target.gz')
ravdess_numeric_labels = joblib.load('./models/ravdess_target.gz')

### Train test split
x_train,x_test,y_train,y_test= train_test_split(np.array(ravdess_data),
                                                ravdess_target,
                                                stratify=ravdess_numeric_labels,
                                                test_size=0.1, random_state=123)

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

x_train.shape

### Lstm
model = Sequential()
model.add(layers.LSTM(128,return_sequences=False,input_shape=(40,1)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(8,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
train_hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,shuffle=True)

### Plot model 
plot_model(model,".models/lstm_rnn2_architecture.png",show_shapes=True,rankdir='LR')


### loss plots using LSTM model
loss = train_hist.history['loss']
val_loss = train_hist.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss,marker='.',linestyle='-', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, '.',linestyle='-', label='Pérdida de valdación')
#plt.title('Training and validation loss')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.tight_layout()
plt.legend()
plt.show()

### accuracy plots
acc = train_hist.history['accuracy']
val_acc = train_hist.history['val_accuracy']
plt.plot(epochs, acc,marker='.',linestyle='-', label='Precisión Entrenamiento')
plt.plot(epochs, val_acc,marker='.',linestyle='-', label='Precisión Validación')
#plt.title('')
plt.xlabel('Iteraciones')
plt.ylabel('Precisión')
plt.legend()
plt.tight_layout()
plt.show()


