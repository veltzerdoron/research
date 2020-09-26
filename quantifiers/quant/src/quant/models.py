from quant.quantifiers import Quantifier

import numpy as np

# keras
from keras.utils import plot_model
from keras.utils import np_utils
from keras.models import clone_model


# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from copy import copy
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = self.__class__.__name__

        # generate the model and plot its diagram
        self._model = None
        self.build()
        plot_model(self._model, to_file='{name}.png'.format(name=self._name), show_shapes=True)

    @abstractmethod
    def build(self):
        """ this is a filler method to put in construction of the actual learning model """
        pass

    def clone(self):
        """ clones the model (wrapping the TF method) """
        clone = copy(self)
        clone._model = clone_model(self._model)


class Classifier(Model, metaclass=ABCMeta):
    """ Classifier model """
    
    def __init__(self, quantifiers, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self._quantifiers = quantifiers

        # encoder to encode quantifier name class values as integers
        self._encoder = LabelEncoder().fit([quantifier.name()
                                            for quantifier in self._quantifiers])
        print("{name} model classifies {classes}".format(name=self._name,
                                                         classes=self._encoder.classes_))

    def generate_labeled_scenes(self, teacher=None,
                                scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len):
        """
        generates and returns scenes and labels

        :param teacher: when provided we use the teach to classify the scenes for the student
        otherwise labeling is done by the quantifier that generates the scene
        :param scene_num: number of scenes to be generated per quantifier

        :param min_len: minimal number of symbols per scene (only includes symbols for elements of a)
        :param max_len: maximal number of symbols per symbols (only includes symbols for elements of a)

        :return: scenes, labels (shuffled and coordinated)
        """

        scenes = np.vstack([quantifier.generate_quantified_scenes(scene_num, min_len, max_len)
                            for quantifier in self._quantifiers])

        # if a model was supplied use it to label the scenes
        if teacher:
            # let the teacher label the scenes
            labels = teacher.predict(scenes)
        else:
            # otherwise build encoder from quantifier names and encode class values as integers
            indices = np.concatenate([[quantifier.name()] * scene_num
                                      for quantifier in self._quantifiers]).ravel()
            labels = self._encoder.transform(indices)

        def tandem_shuffle(a, b):
            """ shuffle the two given arrays in tandem"""
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        # shuffle scenes and labels in tandem
        return tandem_shuffle(scenes, labels)

    def fit(self, scenes, labels, batch_size=20, epochs=50):
        """
        learn the given scenes to labels correlation

        :param scenes: scene inputs
        :param labels: expected outputs

        :param batch_size: batch size passed to the model's fit method
        :param epochs: epoch number passed to the model's fit method
        """

        self._model.fit(scenes, np_utils.to_categorical(labels),
                        batch_size=batch_size, epochs=epochs, verbose=1)

    def predict(self, scenes):
        """
        predict the model results for given scenes

        :param scenes: input scenes to label
        :return: predicted labels
        """

        return np.argmax(self._model.predict(scenes), axis=1)

    def test(self, scenes, labels):
        """ 
        test the model on given labeled scenes
        prints evaluation metrics, confusion matrix and per class classification_report

        :param scenes: scene inputs for testing
        :param labels: expected outputs
        """

        print("Evaluation metrics: ")
        print(self._model.evaluate(scenes, np_utils.to_categorical(labels)))
        results = self.predict(scenes)
        print("Confusion matrix: ")
        print(confusion_matrix(labels, results))
        print("Classification report: ")
        print(classification_report(labels, results, target_names=self._encoder.classes_, digits=4))
        
    def test_random(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len):
        """
        tests the model on randomly generated scenes

        :param scene_num: number of random scenes to test the classifier on
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)
        """
        scenes = Quantifier.generate_random_scenes(scene_num, min_len, max_len)

        # get the models' classifications of the scenes
        results = self.predict(scenes)
        # see if the quantifiers agree with the classifier on the scenes
        print("Quantifier counts: ", np.bincount(results))
        support = sum([any([quantifier.quantify(scene)
                            for quantifier in self._quantifiers])
                       for scene in scenes])
        if support > 0:
            print("Support: ", support)
            print("Accuracy: ", sum([self._quantifiers[result].quantify(scene) 
                                     for result, scene in zip(results, scenes)]) / support)
        else:
            print("NO SUPPORT")


class AE(Model, metaclass=ABCMeta):
    """ Auto encoder model """

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        print("{name} model classifies".format(name=self._name))
