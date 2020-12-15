import unittest

from quants.quantifiers import *
from quants.classifiers import Classifier

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten
from keras.utils import np_utils

# Classifier model for testing


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
                        activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model

    @staticmethod
    def prepare(scenes):
        return np_utils.to_categorical(scenes)


natural_quantifiers = [The(), Both(), No(), All(), Some(), Most()]


class TestClassifier(unittest.TestCase):
    def test_classifier(self):
        CNNClassifier(natural_quantifiers).learn(epochs=1, max_len=200, verbose=0)


if __name__ == '__main__':
    unittest.main()
