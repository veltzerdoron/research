# Classifier unit test

# imports

import unittest

# my class imports

from quants.quantifiers import *
from quants.models import Classifier

# Global imports

# keras and TF imports

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten


from keras.utils import np_utils


class CNNClassifier(Classifier):
    """ Convolutional classifier model builder method """

    def build(self):
        model = Sequential()
        model.add(Conv1D(filters=2, kernel_size=1,
                         use_bias=False,
                         input_shape=(Quantifier.scene_len, len(symbols)), name="conv_1"))
        model.add(Dropout(0.25, name="dropout_1"))
        model.add(Flatten())
        model.add(Dense(len(self._quantifiers),
                        # kernel_initializer="constant", trainable=False, use_bias=False,
                        activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model

    def prepare(self, scenes):
        return np_utils.to_categorical(scenes)


natural_quantifiers = [The(), Both(), No(), All(), Some(), Most()]


class TestClassifier(unittest.TestCase):
    def test_CNN(self):
        CNNClassifier(natural_quantifiers).teach(epochs=3, max_len=100, verbose=1)


if __name__ == '__main__':
    unittest.main()
