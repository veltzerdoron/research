# Auto Encoder (AE) approach

# imports

import unittest

# my class imports
from keras.layers import MaxPooling1D, UpSampling1D
from keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.models import Sequential

from quants.quantifiers import *
from quants.classifiers import SingleLabelClassifier, MultiLabelClassifier, CNNClassifier, AEClassifier

# keras and TF imports
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import *

print("TensorFlow version: ", tf.__version__)
print(tf.config.list_physical_devices(device_type='GPU'))
print(tf.config.list_logical_devices())
print("Keras backend: ", tf.compat.v1.keras.backend.backend())

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# tf.compat.v1.keras.backend.set_session(sess)


class CNNAEClassifier(AEClassifier, CNNClassifier, SingleLabelClassifier):
    """
    CNN Auto Encoder based classifier
    classifies a single quantifier
    """
    def build(self):
        """ Convolutional classifier model builder method """
        model = []
        # build the same AE model for each quantifier
        for _ in self._quantifier_names:
            qmodel = Sequential()
            # encoding
            qmodel.add(Input(name='input', shape=(Quantifier.scene_len, len(symbols))))
            qmodel.add(Conv1D(filters=100, kernel_size=1,
                             trainable=False,
                             use_bias=False,
                             name='conv'))
            qmodel.add(MaxPooling1D(pool_size=2, padding='same'))
            # qmodel.add(Conv1D(100, 5, padding='same', activation='relu'))

            # qmodel.add(MaxPooling1D(pool_size=2, padding='same'))
            # decoding
            # qmodel.add(Conv1DTranspose(100, 1, padding='same', activation='relu'))
            # qmodel.add(UpSampling1D(2))
            # qmodel.add(Conv1DTranspose(100, 5, padding='same', activation='relu'))
            qmodel.add(UpSampling1D(2))
            qmodel.add(Conv1D(filters=len(symbols), kernel_size=1, padding='same', activation='sigmoid'))
            qmodel.compile(loss='mse', optimizer='adam')
            model.append(qmodel)

        return model


class DenseAEClassifier(AEClassifier, MultiLabelClassifier):
    """
    Dense Auto Encoder based classifier
    classifies a single quantifier
    """

    def build(self):
        """ Dense classifier model builder method """
        model = []
        # build the same AE model for each quantifier
        for _ in self._quantifier_names:
            # encoding
            qmodel = Sequential()
            qmodel.add(Input(name='input', shape=Quantifier.scene_len))
            qmodel.add(Dense(250, name="dense2", activation='relu'))
            qmodel.add(Dense(150, name="dense3", activation='sigmoid'))
            # model.add(Dense(50, name="dense4", activation='sigmoid'))
            # decoding
            # model.add(Dense(150, name="dense5", activation='sigmoid'))
            qmodel.add(Dense(250, name="dense6", activation='relu'))
            qmodel.add(Dense(Quantifier.scene_len, name="dense8", activation='relu'))

            # inputs outputs
            qmodel.compile(loss='mse', optimizer='adam')
            model.append(qmodel)

        return model


class CNNSoftmaxClassifier(CNNClassifier, SingleLabelClassifier):
    """
    Convolutional softmax classifier class
    classifies list of quantifiers
    """
    def build(self):
        const_initializer = tf.keras.initializers.Constant(1.)
        # input layer
        scene = Input(name='input', shape=(Quantifier.scene_len, len(symbols)))
        # conv
        conv = Conv1D(filters=self._num_kernels, kernel_size=1,
                      kernel_initializer=const_initializer,
                      trainable=False,
                      use_bias=False,
                      name='conv')(scene)
        # split the
        splitters = tf.split(conv, self._num_kernels, axis=2, name='split')
        # flats
        flats = [Flatten(name='flat_{i}'.format(i=i))(splitters[i])
                 for i in range(self._num_kernels)]
        # dropouts after convolutions
        dropouts = [Dropout(rate=0.15, name='dropout_{i}'.format(i=i))(flats[i])
                    for i in range(self._num_kernels)]

        # single neuron summarizers
        denses = [Dense(1,
                        kernel_initializer=const_initializer,
                        use_bias=False,
                        trainable=False,
                        # activation='relu',
                        name='dense_{i}'.format(i=i))(dropouts[i])
                  for i in range(self._num_kernels)]
        # merge feature extractors
        merge = tf.concat(denses, axis=1, name='concatenate')
        # softmax layer
        softmax = Dense(len(self._quantifier_names),
                        kernel_initializer=const_initializer,
                        use_bias=False,
                        trainable=True,
                        activation='softmax', name="softmax")(merge)
        # inputs outputs
        model = Model(inputs=scene, outputs=softmax)
        # set weights
        conv = model.get_layer('conv')
        conv.set_weights([np.array([self._kernels]).transpose().reshape(1, 4, self._num_kernels)])
        print(conv.get_weights())
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model


class CNNMultiLabelClassifier(CNNClassifier, MultiLabelClassifier):
    """
    Convolutional MultiLabel classifier class
    classifies list of quantifiers
    """
    def build(self):
        const_initializer = tf.keras.initializers.Constant(1.)
        # input layer
        scene = Input(name='input', shape=(Quantifier.scene_len, len(symbols)))
        # conv
        conv = Conv1D(filters=self._num_kernels, kernel_size=1,
                      kernel_initializer=const_initializer,
                      trainable=False,
                      use_bias=False,
                      name='conv')(scene)
        # split the
        splitters = tf.split(conv, self._num_kernels, axis=2, name='split')
        # flats
        flats = [Flatten(name='flat_{i}'.format(i=i))(splitters[i])
                 for i in range(self._num_kernels)]
        # dropouts after convolutions
        dropouts = [Dropout(rate=0.15, name='dropout_{i}'.format(i=i))(flats[i])
                    for i in range(self._num_kernels)]

        # single neuron summarizers
        denses = [Dense(1,
                        kernel_initializer=const_initializer,
                        use_bias=False,
                        trainable=False,
                        # activation='relu',
                        name='dense_{i}'.format(i=i))(dropouts[i])
                  for i in range(self._num_kernels)]
        # merge feature extractors
        merge = tf.concat(denses, axis=1, name='concatenate')
        # multi-label layer
        multilabel = Dense(len(self._quantifier_names),
                           kernel_initializer=const_initializer,
                           use_bias=False,
                           trainable=True,
                           activation='sigmoid', name="multi-label")(merge)
        # inputs outputs
        model = Model(inputs=scene, outputs=multilabel)
        # set weights
        conv = model.get_layer('conv')
        conv.set_weights([np.array([self._kernels]).transpose().reshape(1, len(symbols), self._num_kernels)])
        print(conv.get_weights())
        # compile model
        model.compile(loss='mse', optimizer='adam',
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model


class TestClassifiers(unittest.TestCase):
    def test_CNN_single_label_classifier(self):
        quantifiers = [No(), All(), Some(), Most()]
        # quantifiers = [Between(2, 50), All()]
        kernels = [[1, 0, 0, 0], [1, -1, 0, 0], [0, 1, 0, 0]]
        # kernels = [[1, -1, 0, 0], [0, 1, 0, 0]]
        classifier = CNNSoftmaxClassifier(kernels=kernels, quantifiers=quantifiers)
        classifier.learn(epochs=15, batch_size=1, max_len=100, verbose=1)

    def test_CNN_multi_label_classifier(self):
        quantifiers = [No(), All(), Most(), Some()]
        # quantifiers = [Between(2, 50), All()]
        kernels = [[1, 0, 0, 0], [1, -1, 0, 0], [0, 1, 0, 0]]
        # kernels = [[1, -1, 0, 0], [0, 1, 0, 0]]
        classifier = CNNMultiLabelClassifier(kernels=kernels, quantifiers=quantifiers, other=True)
        classifier.learn(epochs=25, batch_size=1, max_len=100, verbose=1)

    def test_Monotonicity(self):
        most_quantifiers = [Most(), Some()]
        kernels = [[1, -1, 0, 0], [0, 1, 0, 0]]
        CNNSoftmaxClassifier(kernels=kernels,
                             quantifiers=most_quantifiers).learn(epochs=10,
                                                                 verbose=1)

    def test_Every(self):
        all_quantifiers = [All2()]
        kernels = [[1, -1, 0, 0], [0, 1, 0, 0]]
        CNNSoftmaxClassifier(kernels=kernels,
                             quantifiers=all_quantifiers,
                             other=True).learn(epochs=10, batch_size=100,
                                               max_len=100, verbose=1,
                                               contrastive_quantifiers=[Most()])

    def test_Dense_AE_classifier(self):
        DenseAEClassifier(quantifiers=[Most(), All()]).learn(batch_size=1, epochs=10, verbose=1)

    def test_CNN_AE_classifier(self):
        kernels = [[1, -1, 0, 0], [0, 1, 0, 0]]
        CNNAEClassifier(quantifiers=[Most(), All()], kernels=kernels).learn(batch_size=1, epochs=10, verbose=1)


if __name__ == '__main__':
    unittest.main()
