from quants.quantifiers import Quantifier

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import plot_model
from keras.utils import np_utils
from keras.models import clone_model
from sklearn.metrics import confusion_matrix, classification_report

from copy import copy
from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):

    def __init__(self, quantifiers, name=None):
        """
        :param quantifiers: quantifiers to be classified
        """
        self._quantifiers = quantifiers
        self._quantifier_names = [quantifier.name() for quantifier in quantifiers]

        if name:
            self._name = name
        else:
            self._name = self.__class__.__name__

        self._model = self.build()

        print("{name} model classifies {quantifier_names}".format(name=self._name,
                                                                  quantifier_names=self._quantifier_names))

    @abstractmethod
    def build(self):
        """ construction of the actual learning model """
        pass

    def plot(self):
        plot_model(self._model, to_file='{name}.png'.format(name=self._name), show_shapes=True)

    def clone(self):
        """
        clones the model (wrapping the TF method)
        :return: the cloned model
        """
        clone = copy(self)
        clone._model = clone_model(self._model)
        return clone

    @staticmethod
    def prepare(scenes):
        """ override this to conform scenes to network's input, by default scenes are not prepared"""
        return scenes

    def generate_labeled_scenes(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len,
                                teacher=None):
        """
        generates and returns scenes and labels

        :param scene_num: number of scenes to be generated per quantifier
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param teacher: when provided we use the teach to classify the scenes for the student
        otherwise labeling is done by the quantifiers given to the classifier

        :return: scenes, labels (shuffled and coordinated)
        """
        scenes = self.prepare(np.vstack([quantifier.generate_scenes(scene_num, min_len, max_len)
                                         for quantifier in self._quantifiers]))

        if teacher:
            # if a teacher model was supplied let the teacher label the scenes
            labels = teacher.label(scenes)
        else:
            # otherwise encode class names as integer indices
            labels = np.concatenate([[self.index(quantifier.name())] * scene_num
                                     for quantifier in self._quantifiers]).ravel()

        def tandem_shuffle(a, b):
            """ shuffle the two given arrays in tandem"""
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        # shuffle scenes and labels in tandem
        return tandem_shuffle(scenes, labels)

    def learn(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len,
              teacher=None, repeat=1, epochs=50, batch_size=10, verbose=1,
              *vargs, **kwargs):
        """
        This method teaches a classifier to classify its quantifiers

        :param scene_num: number of scenes to be generated per quantifier
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param teacher: when provided we use the teacher to classify the scenes for our student
        otherwise labeling is done by the quantifier that generates the scene (for first teaching iteration)

        :param repeat: teacher student learning performed for repeat # of rounds

        :param epochs: epochs # passed to keras learning
        :param batch_size: batch size passed to keras learning
        :param verbose: verbosity passed to keras methods

        :param vargs : arguments passed on to the model.fit() method
        :param kwargs : key arguments passed on to the model.fit() method

        :return: self (the classifier) after learning
        """
        prev_classifier = teacher
        # with tf.device("/gpu:0"):
        with tf.device("/cpu:0"):
            # iterate while using the previous model as label generator
            for _ in range(repeat):
                # generate fit and test model
                train_scenes_labels = self.generate_labeled_scenes(scene_num, min_len, max_len,
                                                                   teacher=prev_classifier)
                test_scenes_labels = self.generate_labeled_scenes(scene_num,
                                                                  teacher=prev_classifier)
                self.fit(*train_scenes_labels, epochs=epochs, batch_size=batch_size, verbose=verbose, *vargs, **kwargs)
                print("TRAIN")
                self.test(*train_scenes_labels, verbose=verbose)
                print("TEST")
                self.test(*test_scenes_labels, verbose=verbose)
                self.test_random(1000, verbose=verbose)
                prev_classifier = self.clone()
        return self

    @abstractmethod
    def fit(self, scenes, labels,
            *vargs, **kwargs):
        """
        fit the given scenes to their labels

        :param scenes: scene inputs
        :param labels: expected outputs

        :param vargs: arguments passed on to model.fit() method
        :param kwargs: key arguments passed on to model.fit() method
        """
        pass

    @abstractmethod
    def label(self, scenes, verbose=1):
        """
        label the given scenes

        :param scenes: input scenes to label
        :param verbose: verbosity passed to keras methods

        :return: labels
        """
        pass

    @abstractmethod
    def index(self, class_name):
        """
        :param class_name: transform the class name into its class ID
        """
        pass

    def test(self, scenes, labels, verbose=1):
        """
        test the model on given labeled scenes
        prints evaluation metrics, confusion matrix and per class classification_report

        :param scenes: scene inputs for testing
        :param labels: expected outputs

        :param verbose: verbosity passed to keras methods
        """
        print("Evaluation metrics: ")
        print(self._model.evaluate(scenes, np_utils.to_categorical(labels), verbose=verbose))
        results = self.label(scenes, verbose=verbose)
        print("Confusion matrix: ")
        print(pd.DataFrame(
            confusion_matrix(labels, results, labels=self._quantifier_names),
            index=self._quantifier_names,
            columns=self._quantifier_names))
        print("Classification report: ")
        print(classification_report(labels, results, target_names=self._quantifier_names, digits=4))

    def test_random(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len, verbose=1):
        """
        tests the model on randomly generated scenes

        :param scene_num: number of random scenes to test the classifier on
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param verbose: verbosity passed to keras methods
        """
        scenes = self.prepare(Quantifier.generate_random_scenes(scene_num, min_len, max_len))

        # get the models' classifications of the scenes
        results = self.label(scenes, verbose=verbose)
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


class SoftmaxClassifier(Classifier, metaclass=ABCMeta):

    def fit(self, scenes, labels,
            *vargs, **kwargs):
        self._model.fit(scenes, np_utils.to_categorical(labels),
                        *vargs, **kwargs)

    def label(self, scenes, verbose=1):
        predictions = self._model.predict(scenes, verbose=verbose)
        return np.argmax(predictions, axis=1), np.max(predictions, axis=1)


class AEClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, quantifier, *argv, **kwargs):
        """
                :param quantifier: quantifier for generating the input samples to be reconstructed by the AE
        """
        super().__init__(quantifiers=[quantifier], *argv, **kwargs)

        self._threshold = 0

    def fit(self, scenes, labels,
            *vargs, **kwargs):
        # all labels are true here so we ignore them and fit the AE to the scenes
        self._model.fit(scenes, scenes,
                        *vargs, **kwargs)

        # set threshold as twice the average reconstruction error
        output_scenes = self._model.predict(scenes, verbose=0)
        self._threshold = 2 * sum([(np.square(scene - output_output)).mean(axis=0)
                                   for scene, output_output in zip(scenes, output_scenes)]) / len(scenes)

    def label(self, scenes, verbose=1):
        output_scenes = self._model.predict(scenes, verbose=verbose)

        labels, probabilities = [], []
        for scene, output_scene in zip(scenes, output_scenes):
            error = (np.square(scene - output_scene)).mean(axis=0)
            labels.append(error <= self._threshold)
            probabilities.append(error / self._threshold)
        return labels, probabilities

    # def test(self, scenes, labels=None, verbose=1):
    #     """
    #     test the model on given labeled scenes
    #     prints evaluation metrics, confusion matrix and per class classification_report
    #
    #     :param scenes: scene inputs for testing
    #     :param labels: expected outputs
    #
    #     :param verbose: verbosity passed to keras methods
    #     """
    #
    #     if not labels:
    #         labels = [True] * len(scenes)
    #     results = self.predict(scenes, verbose=verbose)
    #     print("Confusion matrix: ")
    #     names = ["Not {}".format(self._quantifier.name()), self._quantifier.name()]
    #     print(pd.DataFrame(
    #         confusion_matrix(labels, results, labels=[True, False]),
    #         index=names,
    #         columns=names))
    #     print("Classification report: ")
    #     print(classification_report(labels, results, labels=[True, False],
    #                                 target_names=names, digits=4))
    #
    # def test_random(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len, verbose=1):
    #     """
    #     tests the model on randomly generated scenes
    #
    #     :param scene_num: number of random scenes to test the classifier on
    #     :param min_len: minimal number of scene symbols (apart from the don't care symbols)
    #     :param max_len: maximal number of scene symbols (apart from the don't care symbols)
    #
    #     :param verbose: verbosity passed to keras methods
    #     """
    #
    #     scenes = self.prepare(np.vstack([self._quantifier.generate_random_scenes(scene_num, min_len, max_len),
    #                                     Quantifier.generate_random_scenes(scene_num, min_len, max_len)]))
    #
    #     # get the models' classifications of the scenes
    #     results = self.predict(scenes, verbose=verbose)
    #     # see if the quantifiers agree with the classifier on the scenes
    #     print("Quantifier counts: ", np.bincount(results))
    #     support = sum([self._quantifier.quantify(scene)
    #                    for scene in scenes])
    #     if support > 0:
    #         print("Support: ", support)
    #         print("Accuracy: ", sum([self._quantifier.quantify(scene) == result
    #                                  for result, scene in zip(results, scenes)]) / support)
    #     else:
    #         print("NO SUPPORT")


class AESoftmaxClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        # build AEClassifier for each of the quantifiers
        self._classifiers = [AEClassifier(quantifier) for quantifier in self._quantifiers]

    def fit(self, scenes, labels,
            *vargs, **kwargs):
        for classifier in self._classifiers:
            classifier.fit(scenes, scenes,
                           *vargs, **kwargs)

    def label(self, scenes, verbose=1):
        probabilities = [classifier.label(scenes)[1] for classifier in self._classifiers]
        # perform softmax on probabilities to get and return labels and probabilities
