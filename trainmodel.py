import numpy as np
import glob
import librosa
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)

SAMPLING_RATE = 44000
FFT_SIZE = 256
STFT_MATRIX_SIZE = 1 + FFT_SIZE // 2

state_names = []
data = np.empty((0, STFT_MATRIX_SIZE), dtype=np.float32)
index = np.empty(0, dtype=np.int32)

model_name = 'wind_nowind.h5'
json_name = 'state_names.json'


for path_name in sorted(glob.glob('data/*/*.wav')):
    state_name = path_name.split('/')[1]
    state_name = state_name.split('_')[0]

    if state_name not in state_names:
        state_names.append(state_name)

    audio, sr = librosa.load(path_name, sr=SAMPLING_RATE, duration=30)
    print('{}: {} ({} Hz) '.format(state_name, path_name, sr))
    d = np.abs(librosa.stft(librosa.util.normalize(audio),
                            n_fft=FFT_SIZE, window='hamming'))
    data = np.vstack((data, d.transpose()))
    index = np.hstack([index, [state_names.index(state_name)] * d.shape[1]])
    
    
N_MID_UNITS = 128
n_states = 2

model = Sequential()
model.add(Dense(N_MID_UNITS, activation='sigmoid', input_dim=STFT_MATRIX_SIZE))
model.add(Dense(N_MID_UNITS // 2, activation='sigmoid'))
model.add(Dense(n_states, activation='softmax'))
model.compile(Adam(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from sklearn.model_selection import train_test_split

index_cat = keras.utils.to_categorical(index)
data_train, data_test, index_train, index_test = train_test_split(data, index_cat, test_size=0.2)

from keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor='val_acc',
                            patience=2,
                            verbose=True,
                            mode='auto')

BATCH_SIZE = 8

model.fit(data_train, index_train, epochs=5, batch_size=BATCH_SIZE,
          validation_split=0.4, callbacks=[es_callback])
          

      
print(model.evaluate(data_test, index_test))


import json


with open(json_name, 'w') as file_to_save:
    json.dump(state_names, file_to_save)

model.save(model_name)
