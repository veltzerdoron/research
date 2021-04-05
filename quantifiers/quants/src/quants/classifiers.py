from quants.quantifiers import *

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import clone_model

from keras.utils import np_utils
from sklearn.metrics import *

from copy import copy
from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):

    def __init__(self, quantifiers, other=False, name=None):
        """
        :param quantifiers: quantifiers to be classified
        :param other: should the classifier allow an other category
        :param name: classifier optional name
        """
        self._quantifiers = quantifiers
        self._quantifier_names = [quantifier.name() for quantifier in quantifiers]
        self._other = other
        if self._other:
            self._quantifier_names += ["Other"]

        if name:
            self._name = name
        else:
            self._name = self.__class__.__name__

        self._model = self.build()

        if isinstance(self._model, list):
            for model in self._model:
                print(model.summary())
        else:
            print(self._model.summary())

        print("{name} model classifies {quantifier_names}".format(name=self._name,
                                                                  quantifier_names=self._quantifier_names))

    # methods wrapping TF methods
    def plot(self):
        plot_model(self._model, to_file='{name}.png'.format(name=self._name), show_shapes=True)

    def summary(self):
        return self._model.summary()

    def clone(self):
        clone = copy(self)
        if isinstance(self._model, list):
            clone._model = [clone_model(model) for model in self._model]
        else:
            clone._model = clone_model(self._model)
        return clone

    @abstractmethod
    def build(self):
        """ model construction abstract method """
        pass

    def prepare_scenes(self, scenes):
        """ override this to conform to network's input, by default scenes are passed as is"""
        return scenes

    def prepare_labels(self, labels):
        """ override this to conform to different network's output, by default labels are passed as categorical"""
        return np_utils.to_categorical(labels, num_classes=len(self._quantifier_names))

    @abstractmethod
    def choose_label(self, true_quantifier_indices):
        """
        :param true_quantifier_indices: indices of quantifiers that have a truth value on the scene
        :return: a possible output for the classifier
        """
        pass

    def label(self, scenes, scene_num=Quantifier.scene_num, teacher=None):
        # label the scenes
        if teacher:
            # if a teacher model was supplied let the teacher label the scenes
            labels = teacher.predict(scenes)
        else:
            # otherwise encode class names as integer indices
            labels = np.vstack([np_utils.to_categorical([i] * scene_num, num_classes=len(self._quantifier_names))
                               for i, _ in enumerate(self._quantifiers)])

            if len(scenes) > len(labels):
                contrastive_labels = []
                for scene in scenes[len(labels):]:
                    true_quantifier_indices = [i for i, quantifier in enumerate(self._quantifiers)
                                               if quantifier.quantify(scene)]
                    if true_quantifier_indices:
                        label = self.choose_label(true_quantifier_indices)
                    else:
                        if self._other:
                            label = len(self._quantifier_names) - 1
                        else:
                            label = np.random.choice(list(range(len(self._quantifiers))))
                    contrastive_labels.append(np_utils.to_categorical(label, num_classes=len(self._quantifier_names)))
                labels = np.vstack([labels, np.vstack(contrastive_labels)])
        return labels

    def generate_labeled_scenes(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len,
                                teacher=None, contrastive_quantifiers=None):
        """
        generates and returns scenes and labels

        :param scene_num: number of scenes to be generated per quantifier
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param teacher: when provided we use the teacher to classify the scenes for the student
        otherwise labeling is done by the quantifiers given to the classifier
        :param contrastive_quantifiers: if given used to generate more labeled cases for the classifier to learn from

        :return: scenes, labels (shuffled and coordinated)
        """

        # generate the scenes from both the quantifiers and the contrastive quantifiers
        scenes = np.vstack([quantifier.generate_scenes(scene_num, min_len, max_len)
                            for quantifier in self._quantifiers])

        if contrastive_quantifiers:
            contrastive_scenes = np.vstack([quantifier.generate_scenes(scene_num, min_len, max_len)
                                            for quantifier in contrastive_quantifiers])
            scenes = np.vstack([scenes, contrastive_scenes])

        labels = self.label(scenes, scene_num, teacher)

        def tandem_shuffle(a, b):
            """ shuffle two arrays in tandem"""
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        # shuffle scenes and labels in tandem
        return tandem_shuffle(scenes, labels)

    def learn(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len,
              repeat=1, epochs=50, batch_size=10, verbose=1,
              teacher=None, contrastive_quantifiers=None,
              *vargs, **kwargs):
        """
        This method teaches a classifier to classify its quantifiers

        :param scene_num: number of scenes to be generated per quantifier
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param repeat: teacher student learning performed for repeat # of rounds

        :param epochs: epochs # passed to keras learning
        :param batch_size: batch size passed to keras learning
        :param verbose: verbosity passed to keras methods

        :param teacher: when provided we use the teacher to classify the scenes for our student
        otherwise labeling is done by the quantifier that generates the scene (for first teaching iteration)
        :param contrastive_quantifiers: if given, used to generate more labeled cases for the classifier to learn from

        :param vargs : arguments passed on to the fit() method
        :param kwargs : key arguments passed on to the fit() method

        :return: self (the classifier) after learning
        """
        prev_classifier = teacher
        # with tf.device("/gpu:0"):
        with tf.device("/cpu:0"):
            # iterate while using the previous model as label generator
            for _ in range(repeat):
                # generate fit and test model
                print("TRAIN")
                train_scenes_labels = self.generate_labeled_scenes(scene_num, min_len, max_len,
                                                                   teacher=prev_classifier,
                                                                   contrastive_quantifiers=contrastive_quantifiers)
                self.fit(*train_scenes_labels, epochs=epochs, batch_size=batch_size, verbose=verbose, *vargs, **kwargs)
                self.test(*train_scenes_labels, verbose=verbose)

                print("TEST")
                # we always test with the full scene length and only on the quantifiers we were supposed to learn
                test_scenes_labels = self.generate_labeled_scenes(scene_num,
                                                                  teacher=prev_classifier)
                self.test(*test_scenes_labels, verbose=verbose)
                self.test_random(1000, verbose=verbose)
                prev_classifier = self.clone()
        return self

    def fit(self, scenes, labels,
            *vargs, **kwargs):
        """
        fit the given scenes to their labels

        :param scenes: scene inputs
        :param labels: expected outputs

        :param vargs: arguments passed on to model.fit() method
        :param kwargs: key arguments passed on to model.fit() method
        """
        self._model.fit(self.prepare_scenes(scenes), labels,
                        *vargs, **kwargs)

    def evaluate(self, scenes, labels, verbose=1):
        """
        fit the given scenes to their labels

        :param scenes: scene inputs
        :param labels: expected outputs

        :param scenes: input scenes to label
        :param verbose: verbosity passed to keras methods
        """
        return self._model.evaluate(scenes, labels, verbose=verbose)

    @abstractmethod
    def predict(self, scenes, verbose=1):
        """
        predict label for the given scenes

        :param scenes: input scenes to label
        :param verbose: verbosity passed to keras methods

        :return: labels for the input scenes
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
        prepared_scenes = self.prepare_scenes(scenes)
        print(self.evaluate(prepared_scenes, labels, verbose=verbose))
        results = self.predict(prepared_scenes, verbose=verbose)
        result_labels = [i for i in range(len(self._quantifier_names))
                         if any([result[i] for result in results])]
        result_targets = [self._quantifier_names[label] for label in result_labels]
        if result_targets:
            print("Classification report: ")
            print(classification_report(labels, results,
                                        labels=result_labels,
                                        target_names=result_targets,
                                        digits=num_symbols))
            # return results to allow further testing if necessary
        else:
            print('No classifications available for report')
        return results, result_labels, result_targets

    def test_random(self, scene_num=Quantifier.scene_num, min_len=0, max_len=Quantifier.scene_len, verbose=1):
        """
        tests the model on randomly generated scenes

        :param scene_num: number of random scenes to test the classifier on
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :param verbose: verbosity passed to keras methods
        """
        scenes = QuantityQuantifier().generate_scenes(scene_num, min_len, max_len)

        # get the models' classifications of the scenes
        results = self.predict(self.prepare_scenes(scenes), verbose=verbose)
        # see if the quantifiers agree with the classifier on the scenes
        print("Quantifier counts: ", np.sum(results, axis=0))
        support = sum([any([quantifier.quantify(scene)
                            for quantifier in self._quantifiers])
                       for scene in scenes])
        if support > 0:
            print("Support: ", support)
            # print("Accuracy: ", sum([int(self._quantifiers[label].quantify(scene))
            #                          for label, scene in zip(results, scenes)
            #                          if label < len(self._quantifier_names)]) / scene_num)
        else:
            print("NO SUPPORT")


class CNNClassifier(Classifier, metaclass=ABCMeta):
    """ Classifier class that has a categorical 1 hot input to a 1d CNN layer """
    def __init__(self, kernels, *argv, **kwargs):
        """
        :param kernels: kernels of the convolutional layers
        """
        self._kernels = kernels
        self._num_kernels = len(kernels)
        super().__init__(*argv, **kwargs)

    def prepare_scenes(self, scenes):
        return np_utils.to_categorical(scenes, num_classes=num_symbols)


class SingleLabelClassifier(Classifier, metaclass=ABCMeta):
    """ a categorical 1 hot output generated from a Softmax layer """
    def choose_label(self, true_quantifier_indices):
        """ return a single random quantifier chosen from the true quantifier indices """
        return [np.random.choice(true_quantifier_indices)]

    def predict(self, scenes, verbose=1):
        results = self._model.predict(scenes, verbose=verbose)
        return np_utils.to_categorical(np.argmax(results, axis=1), num_classes=len(self._quantifier_names))

    def test(self, scenes, labels, verbose=1):
        results, result_labels, result_targets = super().test(scenes, labels, verbose)
        print("Confusion matrix: ")
        print(pd.DataFrame(confusion_matrix(np.argmax(labels, axis=1),
                                            np.argmax(results, axis=1),
                                            labels=result_labels),
                           index=result_targets,
                           columns=result_targets))


class MultiLabelClassifier(Classifier, metaclass=ABCMeta):
    """ a multi-labeled classifier """
    def choose_label(self, true_quantifier_indices):
        """ return the entire list of true quantifier indices """
        return [true_quantifier_indices]

    def predict(self, scenes, verbose=1):
        results = self._model.predict(scenes, verbose=verbose)
        return (results > 0.5).astype(int)

    def test(self, scenes, labels, verbose=1):
        results, result_labels, _ = super().test(scenes, labels, verbose)
        print("Confusion matrix: ")
        print(multilabel_confusion_matrix(labels, results, labels=result_labels))


class AEClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, *argv, **kwargs):
        """
        :param quantifier: quantifier for generating the input samples to be reconstructed by the AE
        """
        super().__init__(*argv, **kwargs)
        self._thresholds = [None] * len(self._quantifiers)

    def fit(self, scenes, labels,
            *vargs, **kwargs):
        # train each AE on the scenes that have its appropriate label flag turned on
        for label, quantifier_name in enumerate(self._quantifier_names):
            print("Training auto encoder for scenes of classifier {q}".format(q=quantifier_name))
            label_scenes = np.vstack([scene for scene, scene_label in zip(scenes, labels) if scene_label[label]])
            prepared_scenes = self.prepare_scenes(label_scenes)
            self._model[label].fit(prepared_scenes, prepared_scenes, *vargs, **kwargs)

            # set threshold as twice the average resulting reconstruction error
            output_scenes = self._model[label].predict(prepared_scenes, verbose=0)
            self._thresholds[label] = 2 * np.square(prepared_scenes - output_scenes).mean()

    def evaluate(self, scenes, labels, verbose=1):
        reconstruction = [np.square(scenes - self._model[label].predict(scenes, verbose=verbose)).mean()
                          for label, _ in enumerate(self._quantifier_names)]
        results = self.predict(scenes, verbose=verbose)

        return reconstruction, precision_recall_fscore_support(labels, results)

    def predict(self, scenes, verbose=1):
        predictions = []
        for label, _ in enumerate(self._quantifier_names):
            output_scenes = self._model[label].predict(scenes, verbose=verbose)
            error = (np.square(scenes - output_scenes))
            while len(error.shape) > 1:
                error = error.mean(axis=1)
            predictions.append(np.array(error < self._thresholds[label]).astype(int))
        return np.vstack(predictions).transpose()
