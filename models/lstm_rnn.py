import numpy as np 
import pandas as pd 
import os 
import librosa 
import joblib
#import wave # read and write WAV files
import matplotlib.pyplot as plt 
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
#from tensorflow.keras.optimizers import rmsprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

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

### Early Stoping Callback
early_stopping = EarlyStopping(patience=5,monitor='val_loss',min_delta=1e-16)

### Lstm
model = Sequential()
model.add(layers.LSTM(128,return_sequences=False,input_shape=(40,1)))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(8,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
train_hist = model.fit(x_train,y_train,
                        validation_data=(x_test,y_test),
                        epochs=150,shuffle=True,
                        callbacks=[early_stopping])



### Plot model 
plot_model(model,"./models/lstm_rnn_architecture.png",show_shapes=True,rankdir='LR')

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

### Confussion Matrix
mapping = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fearful",6:"disgust",7:"surprised"}

test_df = pd.DataFrame(x_test.reshape(288,40))
test_df['emotion'] =  y_test.argmax(axis=1)
test_df['emotion'] = test_df['emotion'].replace(mapping) 
test_df['emotion'].value_counts()

prediction = model.predict(x_test)
test_df['emotion_predicted']=prediction.argmax(axis=1)
test_df['emotion_predicted'] = test_df['emotion_predicted'].replace(mapping) 

cmat = pd.crosstab(test_df['emotion'],test_df['emotion_predicted'])

fig , ax = plt.subplots(figsize=(7,5))
sns.heatmap(cmat,cmap="BuGn",linewidths=.5,annot=True,cbar=False,ax=ax)
ax.set_title("")
ax.set_xlabel("Valor Real")
ax.set_ylabel("Predicción")
plt.tight_layout()
plt.savefig("./models/lstm_rnn2_conf_matrix.png")
plt.show()