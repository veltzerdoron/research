# Classifier approach

# Imports

# my class imports

from quants import *
from models import Classifier

# Global imports

# keras and TF imports

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten

# from tensorflow.keras import initializers

print("TensorFlow version: ", tf.__version__)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
print("Keras backend: ", tf.python.keras.backend.backend())
tf.python.keras.backend.set_session(sess)
tf.config.list_logical_devices()


# Classifier models


class DDNNClassifier(Classifier):
    """ deep dense classifier model builder method """

    def build(self):
        model = Sequential()
        model.add(Dense(Quantifier.scene_len, activation="relu", name="input"))
        model.add(Dropout(0.25, name="dropout_1"))
        model.add(Dense(100, activation="relu", name="dense_2"))
        model.add(Dropout(0.25, name="dropout_2"))
        model.add(Dense(50, activation="relu", name="dense_3"))
        model.add(Dropout(0.25, name="dropout_3"))
        model.add(Dense(len(self._quantifiers), activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model


class DNNClassifier(Classifier):
    """ dense classifier model builder method """

    def build(self):
        model = Sequential()
        model.add(Dense(Quantifier.scene_len, activation="relu", name="input"))
        model.add(Dropout(0.5, name="dropout_1"))
        model.add(Dense(len(self._quantifiers), activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model, False


class CNNClassifier(Classifier):
    """ convolutional neural network """

    def build(self):
        model = Sequential()
        model.add(Conv1D(filters=2, kernel_size=1,
                         use_bias=False,
                         input_shape=(Quantifier.scene_len, len(symbols)), name="conv_1"))
        model.add(Dropout(0.5, name="dropout_1"))
        model.add(Flatten())
        model.add(Dense(len(self._quantifiers),
                        #                         kernel_initializer="constant", trainable=False, use_bias=False,
                        activation='softmax', name="softmax_1"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),
                                                                                  tf.keras.metrics.Recall()])
        return model


# Quantifier sets for learning

natural_quantifiers = [The(), Both(), No(), All(), Some(), Most()]

unnatural_quantifiers = [MinMax(2, 10), MinMax(3, 6), Or([MinMax(2, 5), MinMax(10, 20)])]


# unnatural_quantifiers = [MinMax(2, 5), MinMax(8, 10), MinMax(12, 15), MinMax(17, 20), MinMax(24, 30), MinMax(37, 50)]

def teach(classifier, min_len=0, max_len=Quantifier.scene_len, repeat=1, epochs=50, batch_size=10):
    """
    This method teaches a classifier to classify its quantifiers

    repeat: teacher student learning for repeat # of rounds
    epochs, batch_size: parameters passed to tensorflow learning
    min_len, max_len: generated scene length limits for training (to test generalization)
    """
    last_classifier = None
    with tf.device("/cpu:0"):
        #     with tf.device("/gpu:0"):
        # iterate while using the previous model as label generator
        for _ in range(repeat):
            # generate fit and test model
            if last_classifier:
                train_scenes_labels = classifier.generate_labeled_scenes(last_classifier, min_len, max_len)
                test_scenes_labels = classifier.generate_labeled_scenes(last_classifier)
            else:
                train_scenes_labels = classifier.generate_labeled_scenes(min_len, max_len)
                test_scenes_labels = classifier.generate_labeled_scenes()
            classifier.fit(*train_scenes_labels, epochs=epochs, batch_size=batch_size)
            classifier.test(*test_scenes_labels)
            classifier.test_random(1000)
            last_classifier = classifier.clone()
        return classifier


natural_classifier = teach(Classifier(natural_quantifiers, CNNClassifier), epochs=50, max_len=100)
# natural_classifier = teach(Classifier(natural_quantifiers, DNNClassifier), epochs=500, repeat=3)

# unnatural_model = teach(Classifier(unnatural_quantifiers, CNNClassifier), epochs=50, max_len=100)
unnatural_model = teach(Classifier(unnatural_quantifiers, DNNClassifier), epochs=50, max_len=100)