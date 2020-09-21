import quants as q
import numpy as np

# keras
from keras.utils import plot_model
from keras.utils import np_utils
from keras.models import clone_model

from copy import copy

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

class Classifier():
    
    def __init__(self, quantifiers, builder, name=None):
        self._quantifiers = quantifiers

        if name:
            self._name = name
        else:
            self._name = builder.__name__

        # generate the model and plot its diagram
        self._model, self._one_hot = builder(self._quantifiers)
        plot_model(self._model, to_file='{name}.png'.format(name=self._name), show_shapes=True)

        # encoder to encode class values as integers
        self._encoder = LabelEncoder().fit([quantifier.name()
                                            for quantifier in self._quantifiers])
        print("{name} model classifies {classes}".format(name=self._name,
                                                         classes=self._encoder.classes_))

    def clone(self):
        clone = copy(self)
        clone._model = clone_model(self._model)
        
    def generate_labeled_scenes(self, teacher=None,
                                scene_num=q.scene_num, min_len=0, max_len=q.scene_len):
        """
        generates and returns scenes and labels
        
        scene_num: number of scenes to be generated
        teacher: when provided classifies the scenes (otherwise labeling is done by the quantifiers)

        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)
        """
        
        scenes = np.vstack([q.generate_quantified_scenes(quantifier, scene_num, min_len, max_len)
                            for quantifier in self._quantifiers])
        
        if self._one_hot:
            scenes = np_utils.to_categorical(scenes)
            
        # if a model was supplied use it to label the scenes
        if teacher:
            # let the teacher label the scenes
            labels = teacher.predict(scenes)
        else:
            # encode class values as integers
            indices = np.concatenate([[quantifier.name()] * scene_num
                                      for quantifier in self._quantifiers]).ravel()
            labels = self._encoder.transform(indices)
        # shuffle the scenes and labels in tandem
        def tandem_shuffle(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        return tandem_shuffle(scenes, labels)
        
    # learn the given scenes to labels correlation
    def fit(self, scenes, labels, batch_size=20, epochs=50):
        self._model.fit(scenes, np_utils.to_categorical(labels),
                        batch_size=batch_size, epochs=epochs, verbose=1)
    # predict the model results for given scenes
    def predict(self, scenes):
        return np.argmax(self._model.predict(scenes), axis=1)

    def test(self, scenes, labels):
        """ 
        test the model on given labeled scenes 
        prints evaluation metrics, confusion matrix and per class classification_report
        """

        print("Evaluation metrics: ")
        print(self._model.evaluate(scenes, np_utils.to_categorical(labels)))
        results = self.predict(scenes)
        print("Confusion matrix: ")
        print(confusion_matrix(labels, results))
        print("Classification report: ")
        print(classification_report(labels, results, target_names=self._encoder.classes_, digits=4))
        
    
    def test_random(self, scene_num=q.scene_num, min_len=0, max_len=q.scene_len):
        """
        tests the model on randomly generated scenes
        
        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)
        """
        scenes = q.generate_random_scenes(scene_num, min_len, max_len)
        if self._one_hot:
            scenes = np_utils.to_categorical(scenes)

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