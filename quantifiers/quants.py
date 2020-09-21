import numpy as np

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

# this needs to be a constant value since all models use this to set the actual input width
scene_len = 500

class Quantifier(metaclass=ABCMeta):
    def name(self):
        return "{name}({arguments})".format(
            name = self.__class__.__name__,
            arguments=','.join(["{key}={value}".format(key=key, value=value)                                                        for key,value in self.__dict__.items()]))

    # scene generative methods
    def generate(self, min_len=0, max_len=scene_len):
        """
        Generate a scene that gives a truth value under the quantifer,
        
        This could have been left as an abstract method but we actually have a way to generate a
        prototypical scene without requiring the specific quantifier to define a scene generative
        method. By default we generate a scene prototype by generating a random scene till it passes
        the quantifier method.
        
        NOTE: You should refrain from using this default behavior in general since generating a 
        quantifier matching scene by random might take arbitrary long time depending on the 
        prevalence of its truth value
        
        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)
        
        returns: the generated scene (list of symbols)
        """
        while True:
            scene = self.generate_random_scene(min_len, max_len)
            if self.quantifier(scene):
                return scene
    
    @classmethod
    def generate_random_scene(cls, min_len=0, max_len=scene_len):
        """
        Generate random scene

        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)        

        returns: the generated scene (list of symbols)
        """
        return np.random.permutation(
            np.concatenate([np.random.choice(symbols[:3], min_len),
                            np.random.choice(symbols, max_len - min_len),
                            np.array([symbols[3]] * (scene_len - max_len))]))
        
    def fill(self, counts, min_len=0, max_len=scene_len):
        """ 
        Here we fill in missing counts arbitrarily and then pad the rest with the c_symbol.
        we assume the given counts meet with the quantifier's requirements,
        in natural quantifiers this arisses when:
            - the a_b symbol count is left out, since some quantifiers are interested only in the 
            number of elements in both the a and b sets (s.a. the quantifier 3, both, the, etc...)
            - the b_a symbol count is left out, natural quantifiers are unaffected by this count 
            (this property is called the 'conservative' property of quantifiers in the literature)
            - the c symbol count is left out as these are elements not pertaining to the quantifier

        counts: the input counts
        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)

        returns: the filled in counts (filling in is done in place so this might be disregarded)
        """

        # assert counts don't sum to more than the scene length and
        # apart from don't care which shouldn't be counted scene is within min max length constraints
        sum_counts = sum(counts.values())
        assert(sum_counts >= min_len and sum_counts <= scene_len and sum_counts <= max_len)
        
        # TODO: multinomial distribution is better but with filling in it should't really matter
        for symbol in symbols:
            if symbol not in counts:
                if symbol == c_symbol:
                    # assert again that we haven't generate a scene more that max_len length
                    assert(sum(counts.values()) <= max_len)
                    # fill the rest with don't cares
                    counts[symbol] = scene_len - sum(counts.values())
                else:
                    counts[symbol] = np.random.randint(0, max_len - sum(counts.values()) + 1)

        # now counts should obviously sum to scene length
        assert(sum(counts.values()) == scene_len)

        # apart from the don't care symbol scene should be in the min max length constraints
        care_symbol_len = sum(counts.values()) - counts[c_symbol]
        assert(care_symbol_len <= max_len)
        
        return counts

    def scene(self, counts, min_len=0, max_len=scene_len):
        """
        counts: a count dictionary for each symbol and returns a random scene with 
        ab_len symbols marked 0 representing elements in A and B
        a_b_len symbols marked 1 representing elements in A but not in B.
        b_a_len symbols marked 2 representing elements in B but not in A.
        c_len symbold for the rest of the scene marked 3 meaning irrelevant (complement) elements
        first the symbols are put in order, generating a scene protoype
        and then the prototype is permuted to generate the final scene
        
        counts: the input counts
        min_len: minimal number of scene symbols (apart from the don't care symbols)
        max_len: maximal number of scene symbols (apart from the don't care symbols)

        returns: the constructed scene
        """
        # make sure we received legal parameters
        assert(all([count >= 0 for count in counts]))  # assert counts are non negative
        if len(counts.keys()) == len(symbols):
            # if all counts accounted for check the sum to scene_len
            assert(sum(counts.values()) == scene_len)
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

## Natural Quantifiers

# Most quantifier
class Most(Quantifier):
    def generate(self, min_len=0, max_len=scene_len):
        # Here we assume implicatures, namely, if more than one element in A&B 'most' excludes 'all'
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

# Some quantifier
class Some(Quantifier):
    def generate(self, min_len=0, max_len=scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(1, min_len), max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 0

# Few quantifier
class Few(Quantifier):
    def generate(self, min_len=0, max_len=scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(2, min_len), max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 1

# No quantifier
class No(Quantifier):
    def generate(self, min_len=0, max_len=scene_len):
        # here we assume some bs exist otherwise saying "No as are bs" is quite strange
        counts = defaultdict(int)
        counts[ab_symbol] = 0
        counts[a_b_symbol] = np.random.randint(max(1, min_len), high=max_len + 1)
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] == 0 and counts[a_b_symbol] > 0

# All quantifier
class All(Quantifier):
    def generate(self, min_len=0, max_len=scene_len):
        # here we assume some as exist otherwise saying "All as are bs" is quite strange
        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(max(1, min_len), high=max_len + 1)
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] > 0 and counts[a_b_symbol] == 0
    
# The quantifier
class The(Quantifier):    
    def generate(self, min_len=0, max_len=scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = 1
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == 1 and counts[a_b_symbol] == 0

# Both quantifier
class Both(Quantifier):    
    def generate(self, min_len=0, max_len=scene_len):
        counts = defaultdict(int)
        counts[ab_symbol] = 2
        counts[a_b_symbol] = 0
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == 2 and counts[a_b_symbol] == 0

# N quantifier
class N(Quantifier):
    def __init__(self, n):
        self.n = n
    
    def generate(self, min_len=0, max_len=scene_len):
        assert(min_len <= self.n <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.n, high=max_len + 1)  # n := (>= n) implicature
        return self.scene(counts, min_len, max_len)

    def quantification(self, counts):
        return counts[ab_symbol] >= self.n

# exactly N quantifier
class exactly_N(Quantifier):   
    def __init__(self, n):
        self.n = n
    
    def generate(self, min_len=0, max_len=scene_len):
        assert(min_len <= self.n <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = self.n  # n:= (== n)
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] == self.n

## General Quantifiers

# Min - Max quantifier (requires the min, max members to operate)
class MinMax(Quantifier):
    def __init__(self, m, M):
        assert(m < M)
        self.m = m
        self.M = M
    
    def generate(self, min_len=0, max_len=scene_len):
        assert(min_len <= self.m <= self.M <= max_len)

        counts = defaultdict(int)
        counts[ab_symbol] = np.random.randint(self.m, high=self.M + 1)
        return self.scene(counts, min_len, max_len)
    
    def quantification(self, counts):
        return counts[ab_symbol] >= self.m and counts[ab_symbol] <= self.M

# Conjunction quantifiers (conjoining lists of quantifiers)
class Conjunction(Quantifier):
    def name(self):
        return "{name}({quantifiers})".format(
            name = self.__class__.__name__,
            quantifiers=','.join([quantifier.name() for quantifier in self.quantifiers]))
    
    def __init__(self, quantifiers):
        self.quantifiers = quantifiers
    
    def generate(self, min_len=0, max_len=scene_len):
        return np.random.choice(self.quantifiers).generate(min_len, max_len)

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

def generate_random_scenes(scene_num=scene_num, min_len=0, max_len=scene_len):
    # define input scenes
    # make scenes a matrix for training
    scenes = np.concatenate([Quantifier.generate_random_scene(min_len, max_len) 
                             for _ in range(scene_num)], axis=0)

    # reshape input into [samples, timesteps] features are added as one hot encoded when neccessary
    scenes = scenes.reshape((scene_num, scene_len))
    return scenes

def generate_quantified_scenes(quantifier, scene_num=scene_num, min_len=0, max_len=scene_len):
    # define input scenes
    scenes = [quantifier.generate(min_len, max_len)
              for _ in range(scene_num)]
    # sanity check
    assert(np.all([quantifier.quantify(scene) for scene in scenes]))

    # make scenes a matrix for training
    scenes = np.concatenate(scenes, axis=0)
    # reshape input into [samples, timesteps, features]
    scenes = scenes.reshape((scene_num, scene_len))
    return scenes