import numpy as np

from collections import defaultdict
from abc import ABCMeta, abstractmethod

"""
First we define the base Quantifier class which implements generative and quantification methods 
then we use this class to define some common natural and general (unnatural) quantifiers.
"""

# scene symbols
ab_symbol = 0  # elements in A and in B (A&B)
a_b_symbol = 1  # elements in A but not in B (A-B)
b_a_symbol = 2  # elements in B but not in A (B-A)
c_symbol = 3  # irrelevant (don't care) padding elements

symbols = [ab_symbol, a_b_symbol, b_a_symbol, c_symbol]


class Quantifier(metaclass=ABCMeta):
    """ Quantifier abstract base class """

    scene_len = 500  # if changed value must be constant across usage since models use it to set input width
    scene_num = 1000  # this can be changed arbitrarily

    def name(self):
        return "{name}({arguments})".format(
            name=self.__class__.__name__,
            arguments=','.join(["{key}={value}".format(key=key, value=value)
                                for key, value in self.__dict__.items()]))

    # scene generative methods

    def generate_scene(self, min_len=0, max_len=scene_len):
        """
        Generate a scene that gives a truth value under the quantifier,
        
        This could have been left as an abstract method but we actually have a way to generate a
        prototypical scene without requiring the specific quantifier to define a scene generative
        method. By default we generate a scene prototype by generating a random scene till it passes
        the quantifier method.
        
        NOTE: You should refrain from using this default behavior in general since generating a 
        quantifier matching scene by random might take arbitrary long time depending on the 
        prevalence of its truth value
        
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: the generated scene (list of symbols)

        """
        while True:
            scene = self.generate_random_scene(min_len, max_len)
            if self.quantify(scene):
                return scene

    def generate_scenes(self, scene_num=scene_num, min_len=0, max_len=scene_len):
        """
        Generate scene_num scenes that gives a truth value under the quantifier,

        :param scene_num: number of scenes to generate
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: generated scenes
        """
        scenes = [self.generate(min_len, max_len)
                  for _ in range(scene_num)]
        # sanity check that the quantifier is generated quantifier true scenes
        assert (np.all([self.quantify(scene) for scene in scenes]))

        # generate scenes as a matrix for training
        scenes = np.concatenate(scenes, axis=0)

        # reshape scenes into [scene_num, scene_len] and transform to one hot encoding is necessary
        scenes = scenes.reshape((scene_num, Quantifier.scene_len))
        return scenes

    @staticmethod
    def generate_random_scene(min_len=0, max_len=scene_len):
        """
        Generate random scene

        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: the generated scene (list of symbols)
        """
        return np.random.permutation(
            np.concatenate([np.random.choice(symbols[:3], min_len),
                            np.random.choice(symbols, max_len - min_len),
                            np.array([symbols[3]] * (Quantifier.scene_len - max_len))]))

    @staticmethod
    def generate_random_scenes(scene_num=scene_num, min_len=0, max_len=scene_len):
        """
        generates random scenes

        :param scene_num: number of scenes to generate
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: generated scenes
        """
        scenes = np.concatenate([Quantifier.generate_random_scene(min_len, max_len)
                                 for _ in range(scene_num)], axis=0)

        # reshape scenes into [scene_num, scene_len] and transform to one hot encoding is necessary
        scenes = scenes.reshape((scene_num, Quantifier.scene_len))
        return scenes

    @staticmethod
    def fill(counts, min_len=0, max_len=scene_len):
        """
        
        """"""" 
        Fills in missing counts arbitrarily and then pad the rest with the c_symbol. we assume the given counts meet 
        the quantifier's requirements, in natural quantifiers this arises when:

            - the a_b symbol count is left out, since some quantifiers are interested only in the 
            number of elements in both the a and b sets (s.a. the quantifier 3, both, the, etc...)
            - the b_a symbol count is left out, natural quantifiers are unaffected by this count 
            (this property is called the 'conservative' property of quantifiers in literature)
            - the c symbol count is left out as these are filled in padding symbols

        :param counts: the input counts 
        :param min_len: minimal number of scene symbols (apart from the don't care symbols) 
        :param max_len: maximal number of scene symbols (apart from the don't care symbols) 

        :return: the filled in counts (filling in is done in place so this might be disregarded)
        """

        # assert counts don't sum to more than the scene length and
        # apart from don't care which shouldn't be counted scene is within min max length constraints
        sum_counts = sum(counts.values())
        assert(min_len <= sum_counts <= Quantifier.scene_len and sum_counts <= max_len)
        
        # TODO: multinomial distribution is better (with filling in it shouldn't really matter)
        for symbol in symbols:
            if symbol not in counts:
                if symbol == c_symbol:
                    # assert again that we haven't generate a scene more that max_len length
                    assert(sum(counts.values()) <= max_len)
                    # fill the rest with don't cares
                    counts[symbol] = Quantifier.scene_len - sum(counts.values())
                else:
                    counts[symbol] = np.random.randint(0, max_len - sum(counts.values()) + 1)

        # now counts should obviously sum to scene length
        assert(sum(counts.values()) == Quantifier.scene_len)

        # apart from the don't care symbol scene should be in the min max length constraints
        care_symbol_len = sum(counts.values()) - counts[c_symbol]
        assert(care_symbol_len <= max_len)
        
        return counts

    def scene(self, counts, min_len=0, max_len=scene_len):
        """
        receives a count dictionary for each symbol and returns a random scene with
        ab count symbols marked 0 representing elements in A and B.
        a_b count symbols marked 1 representing elements in A but not in B.
        b_a count symbols marked 2 representing elements in B but not in A.
        c count symbols for the rest of the scene marked 3 meaning irrelevant (complement) elements
        first the symbols are put in order, generating a scene prototype
        then the prototype is permuted to generate the final scene.
        
        :param counts: the input counts
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: the constructed scene
        """
        # parameters sanity check
        assert(all([count >= 0 for count in counts]))  # assert counts are non negative
        if len(counts.keys()) == len(symbols):
            # if all counts accounted for check they sum to scene_len
            assert(sum(counts.values()) == Quantifier.scene_len)
        else:
            # otherwise, assume all important counts given, fill in missing counts arbitrarily
            self.fill(counts, min_len, max_len)

        # generate and return the permuted scene prototype
        prototype = np.concatenate([[int(symbol)] * counts[symbol] 
                                    for symbol in symbols]).astype(int)
        return np.random.permutation(prototype)

    # scene quantification methods

    def quantify(self, scene):
        # evaluate the scene by calling the quantifier with the counts of the different symbols
        return self.quantification(defaultdict(int, zip(*np.unique(scene, return_counts=True))))
    
    @abstractmethod
    def quantification(self, counts):
        pass

# Natural Quantifiers


class Most(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        # assume implicature, i.e. if more than one element in A&B 'most' excludes 'all'
        counts = defaultdict(int)
        scene_len = np.random.randint(max(min_len, 1), max_len + 1)
        if scene_len > 2:
            ab_count = np.random.randint(scene_len // 2 + 1, scene_len)  # + 1 would not exclude all
        else:
            ab_count = scene_len
        counts[ab_symbol] = ab_count
        counts[a_b_symbol] = scene_len - ab_count
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > counts[a_b_symbol]


class Some(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(1, min_len), max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 0


class Few(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(2, min_len), max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 1


class No(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        # assume some bs exist otherwise "?No as are bs" is quite strange
        counts = defaultdict(int)
        counts[ab_symbol] = 0
        counts[a_b_symbol] = np.random.randint(max(1, min_len), high=max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] == 0 and counts[a_b_symbol] > 0


class All(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        # here we assume some as exist otherwise saying "All as are bs" is quite strange
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(1, min_len), high=max_len + 1)
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 0 and counts[a_b_symbol] == 0
    

class The(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = 1
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == 1 and counts[a_b_symbol] == 0


class Both(Quantifier):
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = 2
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == 2 and counts[a_b_symbol] == 0


class N(Quantifier):
    def __init__(self, n):
        self.n = n
    
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        assert(min_len <= self.n <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.n, high=max_len + 1)  # n := (>= n) implicature
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] >= self.n


class ExactlyN(Quantifier):
    def __init__(self, n):
        self.n = n
    
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        assert(min_len <= self.n <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = self.n  # n:= (== n)
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == self.n


class MinMax(Quantifier):
    def __init__(self, mini, maxi):
        assert(mini < maxi)
        self.mini = mini
        self.maxi = maxi
    
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        assert(min_len <= self.mini <= self.maxi <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.mini, high=self.maxi + 1)
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return self.mini <= counts[ab_symbol] <= self.maxi


class Conjunction(Quantifier, metaclass=ABCMeta):
    def name(self):
        return "{name}({quantifiers})".format(
            name=self.__class__.__name__,
            quantifiers=','.join([quantifier.name() for quantifier in self.quantifiers]))
    
    def __init__(self, quantifiers):
        self.quantifiers = quantifiers
    
    def generate(self, min_len=0, max_len=Quantifier.scene_len):
        return np.random.choice(self.quantifiers).generate(min_len, max_len)


class Or(Conjunction):
    def quantification(self, counts):
        return any([quantifier.quantification(counts) for quantifier in self.quantifiers])


class And(Conjunction):
    def quantification(self, counts):
        return all([quantifier.quantification(counts) for quantifier in self.quantifiers])
