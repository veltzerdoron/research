# Auto Encoder (AE) approach

# imports

# my class imports

from quant.quantifiers import *
from quant.models import AE

# keras and TF imports

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling1D, Bidirectional, RepeatVector, SimpleRNN, \
    TimeDistributed, LSTM

# from tensorflow.keras import initializers

print("TensorFlow version: ", tf.__version__)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
print("Keras backend: ", tf.python.keras.backend.backend())
tf.python.keras.backend.set_session(sess)
tf.config.list_logical_devices()


# Auto Encoder  models


class DenseAE(AE):

    def build(self):
        model = Sequential()
        # encoding
        model.add(Dense(75, activation='sigmoid', input_shape=(100, 3)))
        model.add(Flatten())
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(25, activation='sigmoid'))
        # decoding
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(75, activation='sigmoid'))
        model.add(Dense(Quantifier.scene_len, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')
        return model


class CNNAE(AE):

    def build(self):
        model = Sequential()
        # encoding
        model.add(Conv1D(60, 32, strides=1, activation='relu', padding='causal', input_shape=(Quantifier.scene_len, 1)))
        model.add(Conv1D(80, 10, strides=1, activation='relu', padding='causal'))
        model.add(Dropout(0.25))
        model.add(Conv1D(100, 5, strides=1, activation='relu', padding='causal'))
        model.add(MaxPooling1D(1))
        # decoding
        model.add(Dropout(0.25))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        return model


class RNNAE(AE):

    def build(self):

        model = Sequential()
        model.add(Bidirectional(SimpleRNN(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1))))
        model.add(Dropout(0.5))
        # model.add(Bidirectional(SimpleRNN(15, activation='sigmoid', return_sequences=True)))
        # model.add(Dropout(0.5))
        model.add(RepeatVector(Quantifier.scene_len))
        model.add(SimpleRNN(15, activation='sigmoid', return_sequences=True))
        model.add(Dropout(0.5))
        # model.add(Bidirectional(SimpleRNN(25, activation='sigmoid', return_sequences=True)))
        # model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        return model


class LSTMAE(AE):

    def build(self):
        model = Sequential()
        model.add(LSTM(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1)))
        model.add(Dropout(0.5))
        model.add(RepeatVector(Quantifier.scene_len))
        model.add(LSTM(15, activation='sigmoid', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        return model


class BLSTMAE(AE):

    def build(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1))))
        model.add(Dropout(0.5))
        model.add(RepeatVector(Quantifier.scene_len))
        model.add(Bidirectional(LSTM(15, activation='sigmoid', return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        return model
