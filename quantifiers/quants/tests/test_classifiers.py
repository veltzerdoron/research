# Auto Encoder (AE) approach

# imports

import unittest

# my class imports

from quants.quantifiers import *
from quants.classifiers import SoftmaxClassifier, AEClassifier

# keras and TF imports

import tensorflow as tf

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils

# print("TensorFlow version: ", tf.__version__)
#
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# print("Keras backend: ", tf.python.keras.backend.backend())
# tf.python.keras.backend.set_session(sess)
# tf.config.list_logical_devices()

# Auto Encoder Classifier model for testing


class CNNAEClassifier(AEClassifier):
    def build(self):
        """ Convolutional classifier model builder method """
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
        model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                             tf.keras.metrics.Recall()])
        return model

    @staticmethod
    def prepare(scenes):
        return np_utils.to_categorical(scenes)


class DenseAEClassifier(AEClassifier):

    def build(self):
        """ dense classifier model builder method """
        model = Sequential()
        # encoding
        # model.add(Dense(Quantifier.scene_len, use_bias=False, trainable=False,
        #                 weights=(np.eye(Quantifier.scene_len),), activation='linear',
        #                 input_dim=Quantifier.scene_len))
        # model.add(Dense(Quantifier.scene_len, use_bias=False, activation='linear', input_dim=Quantifier.scene_len))
        # model.add(Flatten())
        model.add(Dense(75, activation='sigmoid', input_dim=Quantifier.scene_len))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(25, activation='sigmoid'))
        # decoding
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(75, activation='sigmoid'))
        model.add(Dense(Quantifier.scene_len, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                             tf.keras.metrics.Recall()])
        return model

# class RNNAE(AE):
#     def build(self):
#         model = Sequential()
#         model.add(Bidirectional(SimpleRNN(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1))))
#         model.add(Dropout(0.5))
#         # model.add(Bidirectional(SimpleRNN(15, activation='sigmoid', return_sequences=True)))
#         # model.add(Dropout(0.5))
#         model.add(RepeatVector(Quantifier.scene_len))
#         model.add(SimpleRNN(15, activation='sigmoid', return_sequences=True))
#         model.add(Dropout(0.5))
#         # model.add(Bidirectional(SimpleRNN(25, activation='sigmoid', return_sequences=True)))
#         # model.add(Dropout(0.5))
#         model.add(TimeDistributed(Dense(1)))
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
#
# class LSTMAE(AE):
#     def build(self):
#         model = Sequential()
#         model.add(LSTM(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1)))
#         model.add(Dropout(0.5))
#         model.add(RepeatVector(Quantifier.scene_len))
#         model.add(LSTM(15, activation='sigmoid', return_sequences=True))
#         model.add(Dropout(0.5))
#         model.add(TimeDistributed(Dense(1)))
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
#
# class BLSTMAE(AE):
#
#     def build(self):
#         model = Sequential()
#         model.add(Bidirectional(LSTM(15, activation='sigmoid', input_shape=(Quantifier.scene_len, 1))))
#         model.add(Dropout(0.5))
#         model.add(RepeatVector(Quantifier.scene_len))
#         model.add(Bidirectional(LSTM(15, activation='sigmoid', return_sequences=True)))
#         model.add(Dropout(0.5))
#         model.add(TimeDistributed(Dense(1)))
#         model.compile(optimizer='adam', loss='mse')
#         return model

# CNN Softmax Classifier model for testing


class CNNSoftmaxClassifier(SoftmaxClassifier):
    def build(self):
        """ Convolutional classifier model builder method """
        model = Sequential()
        model.add(Conv1D(filters=2, kernel_size=1,
                         use_bias=False,
                         input_shape=(Quantifier.scene_len, len(symbols)), name="conv_1"))
        # model.add(Dropout(0.50, name="dropout_1"))
        model.add(Flatten())
        model.add(Dense(len(self._quantifiers),
                        activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model

    @staticmethod
    def prepare(scenes):
        return np_utils.to_categorical(scenes)


class TestClassifiers(unittest.TestCase):
    def test_CNN_softmax_classifier(self):
        natural_quantifiers = [No(), All(), Some(), Most()]
        CNNSoftmaxClassifier(natural_quantifiers).learn(epochs=100, verbose=1).plot()

    def test_Monotonicity(self):
        natural_quantifiers = [AFew(), Between(3, 20)]
        CNNSoftmaxClassifier(natural_quantifiers).learn(epochs=100, verbose=1).plot()

    def test_Dense_AE_classifier(self):
        quantifier = Most()
        DenseAEClassifier(quantifier).learn(epochs=100, verbose=1).plot()


if __name__ == '__main__':
    unittest.main()
