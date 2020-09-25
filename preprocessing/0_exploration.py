import librosa
import os
import matplotlib.pyplot as plt
import librosa.display

path = './data/OAF_angry/'
files = os.listdir(path)

x, sr = librosa.load(path+files[0])

plt.figure(figsize=(20,5))
librosa.display.waveplot(x,sr=sr)
plt.show()

