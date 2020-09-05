import numpy as np
import pandas as pd
import inspect

from collections import defaultdict
from abc import ABCMeta, abstractmethod

"""
First we define the base Quantifier class which implements generative and quantification methods, then we use this class to define some common natural and general (unnatural) quantifiers.
"""

# Quantifier abstract base class
# define 3 symbols for, member in A and B member in b but not a and irrelevant item scenes
ab_symbol = 0  # elements in A and in B (A&B)
a_b_symbol = 1  # elements in A but not in B (A-B)
b_a_symbol = 2  # elements in B but not in A (B-A)
c_symbol = 3  # irrelevant elements

symbols = [ab_symbol, a_b_symbol, b_a_symbol, c_symbol]

scene_len = 500

class Quantifier(metaclass=ABCMeta):
    def name(self):
        return "{name}({arguments})".format(
            name = self.__class__.__name__,
            arguments=','.join(["{key}={value}".format(key=key, value=value)                                                        for key,value in self.__dict__.items()]))

    # scene generative methods
    def generate(self):
        """
        Generate a scene that gives a truth value under the quantifer,
        
        This could have been left as an abstract method but we actually have a way to generate a
        prototypical scene without requiring the specific quantifier to define a scene generative
        method. By default we generate a scene prototype by generating a random scene till it passes
        the quantifier method.
        
        NOTE: You should refrain from using this default behavior in general since generating a 
        quantifier matching scene by random might take arbitrary long time depending on the 
        prevalence of its truth value
        """
        while True:
            scene = self.generate_random_scene(scene_len)
            if self.quantifier(scene):
                return scene
    
    @classmethod
    def generate_random_scene(cls):
        """
        Generate a completely random scene
        """
        return np.random.choice(symbols, scene_len)
        
    def fill(self, counts):
        """ 
        Here we fill in missing counts arbitrarily and then pad the rest with the c_symbol.
        we assume the given counts meet with the quantifier's requirements,
        in natural quantifiers this arisses when:
            - the a_b symbol count is left out, since some quantifiers are interested only in the 
            number of elements in both the a and b sets (s.a. the quantifier 3, both, the, etc...)
            - the b_a symbol count is left out, natural quantifiers are unaffected by this count 
            (this property is called the 'conservative' property of quantifiers in the literature)
            - the c symbol count is left out as these are elements not pertaining to the quantifier
        counts: the given filled in counts
        """
        assert(sum(counts.values()) <= scene_len)
        # TODO : use multinomial distribution
        for symbol in symbols:
            if symbol not in counts:
                if symbol == c_symbol:
                    # fill the rest with don't care
                    counts[symbol] = scene_len - sum(counts.values())
                else:
                    counts[symbol] = np.random.randint(0, scene_len - sum(counts.values()) + 1)

        # now counts should obviously sum to scene_len, assert this
        assert(sum(counts.values()) == scene_len)

    def scene(self, counts):
        """
        counts: a count dictionary for each symbol and returns a random scene with 
        ab_len symbols marked 0 representing elements in A and B
        a_b_len symbols marked 1 representing elements in A but not in B.
        b_a_len symbols marked 2 representing elements in B but not in A.
        c_len symbold for the rest of the scene marked 3 meaning irrelevant (complement) elements
        first the symbols are put in order, generating a scene protoype
        and then the prototype is permuted to generate the final scene
        """
        # make sure we received legal parameters
        assert(all([count >= 0 for count in counts]))  # assert counts are non negative
        if len(counts.keys()) == len(symbols):
            # if all counts accounted for check the sum to scene_len
            assert(sum(counts.values()) == scene_len)
        else:
            # otherwise, assume all important counts given, fill in missing counts arbitrarily
            self.fill(counts)

        # generate the scene prototype and return the permuted prototype
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

## Natural Quantifiers

# Most quantifier
class Most(Quantifier):
    def generate(self):
        # We assume implicature, i.e. if there is more than one element in A&B 'most' excludes 'all'
        counts = defaultdict(int)
        counts[ab_symbol] = ab_count = np.random.randint(1, high=max(scene_len, 2))
        counts[a_b_symbol] = 0 if ab_count == 1 else \
            np.random.randint(1, min(ab_count, scene_len - ab_count + 1))
        return self.scene(counts)

    def quantification(self, counts):
        return counts[ab_symbol] > counts[a_b_symbol]

# Some quantifier
class Some(Quantifier):
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(1, high=scene_len + 1)
        return self.scene(counts)

    def quantification(self, counts):
        return counts[ab_symbol] > 0

# Few quantifier
class Few(Quantifier):
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(2, high=scene_len + 1)
        return self.scene(counts)

    def quantification(self, counts):
        return counts[ab_symbol] > 1

# No quantifier
class No(Quantifier):
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = 0
        counts[a_b_symbol] = np.random.randint(1, high=scene_len + 1)
        return self.scene(counts)

    def quantification(self, counts):
        return counts[ab_symbol] == 0 and counts[a_b_symbol] > 0

# Every quantifier
class Every(Quantifier):
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(1, high=scene_len + 1)
        counts[a_b_symbol] = 0
        return self.scene(counts)

    def quantification(self, counts):
        return counts[a_b_symbol] == 0
    
# Both quantifier (ab == 2)
class Both(Quantifier):    
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = 2
        counts[a_b_symbol] = 0
        return self.scene(counts)
    
    def quantification(self, counts):
        return counts[ab_symbol] == 2 and counts[a_b_symbol] == 0

# N quantifier
class N(Quantifier):
    def __init__(self, n):
        self.n = n
    
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.n, high=scene_len + 1)  # n := (>= n) implicature
        return self.scene(counts)

    def quantification(self, counts):
        return counts[ab_symbol] >= self.n

# exactly N quantifier
class exactly_N(Quantifier):   
    def __init__(self, n):
        self.n = n
    
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = self.n  # n:= (== n)
        return self.scene(counts)
    
    def quantification(self, counts):
        return counts[ab_symbol] == self.n

## General Quantifiers

# Min - Max quantifier (requires the min, max members to operate)
class MinMax(Quantifier):
    def __init__(self, m, M):
        self.m = m
        self.M = M
    
    def generate(self):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.m, high=self.M + 1)
        return self.scene(counts)
    
    def quantification(self, counts):
        return counts[ab_symbol] >= self.m and counts[ab_symbol] <= self.M

# Conjunction quantifiers (conjoining lists of quantifiers)
class Conjunction(Quantifier, metaclass=ABCMeta):
    def name(self):
        return "{name}({quantifiers})".format(
            name = self.__class__.__name__,
            quantifiers=','.join([quantifier.name() for quantifier in self.quantifiers]))
    
    def __init__(self, quantifiers):
        self.quantifiers = quantifiers
    
    def generate(self):
        return np.random.choice(self.quantifiers, 1)[0].generate()
    
    @abstractmethod
    def quantification(self, counts):
        pass

# Or quantifier
class Or(Conjunction):        
    def quantification(self, counts):
        return any([quantifier.quantification(counts) for quantifier in self.quantifiers])

# And quantifier
class And(Conjunction):        
    def quantification(self, counts):
        return all([quantifier.quantification(counts) for quantifier in self.quantifiers])

# Generate random and quantified scenes

scene_num = 1000

def generate_random_scenes(scene_num=scene_num):
    # define input scenes
    # make scenes a matrix for training
    scenes = np.concatenate([Quantifier.generate_random_scene() for _ in range(scene_num)], axis=0)
    # reshape input into [samples, timesteps, features]
    scenes = scenes.reshape((scene_num, scene_len))
    return scenes
    return 1

def generate_quantified_scenes(quantifier, scene_num=scene_num):
    # define input scenes
    scenes = [quantifier.generate() for _ in range(scene_num)]
    # sanity check
    assert(np.all([quantifier.quantify(scene) for scene in scenes]))

    # make scenes a matrix for training
    scenes = np.concatenate(scenes, axis=0)
    # reshape input into [samples, timesteps, features]
    scenes = scenes.reshape((scene_num, scene_len))
    return scenes