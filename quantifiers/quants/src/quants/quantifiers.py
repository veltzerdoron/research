from collections import defaultdict

from quants.constraints import *

import numpy as np
import random

from itertools import chain
from abc import ABCMeta


class Quantifier(metaclass=ABCMeta):
    """
    base abstract Quantifier class which defines all generative and quantification abstract methods to be implemented
    """
    scene_len = 500  # if changed value must be constant across usage since models use it to set input width
    scene_num = 1000  # this can be changed arbitrarily

    def name(self, **kwargs):
        return "{name}({arguments})".format(
            name=self.__class__.__name__,
            arguments=','.join(["{key}={value}".format(key=key, value=value)
                                for key, value in chain(self.__dict__.items(), kwargs.items())]))

    # random methods

    @staticmethod
    def _random_interleave(a, b):
        """
        randomly interleave two lists

        :param a, b: lists to be interleaved
        :return: interleaved list
        """
        return [random.choice([a, b]).pop(0) for _ in range(len(a) + len(b)) if (a and b)] + a + b

    @staticmethod
    def _random_generate_constant_sum(n, s, m=1):
        """
        generate n uniform random variables that sum up to s

        :param n: number of random variables to generate
        :param s: sum of those n random variables
        :param m: number of samples to generate

        :return: m samples of n uniform random variables that sum up to s
        """
        assert (n >= 0) and (s >= 0) and (m >= 1), "Illegal parameters in number generation method"
        if n == 0:
            return []
        uv = np.hstack([np.zeros([m, 1]),
                        np.sort(np.random.randint(low=0, high=s + 1, size=[m, n-1]), axis=1),
                        np.full([m, 1], s)])
        return np.diff(uv, axis=1).astype(int)

    # generative methods

    def generate_scene(self, min_len=0, max_len=scene_len):
        """
        Generate random scene directly via uniform distribution

        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: the generated scene (list of symbols)
        """
        assert (0 <= min_len <= max_len <= Quantifier.scene_len), "Illegal scene length limits given"

        # Note: this is commented out, Multinomial doesn't cover all the possibilities of the counts simplex very well
        # return np.random.permutation(
        #     np.concatenate([np.random.choice(symbols[:3], min_len),
        #                     np.random.choice(symbols, max_len - min_len),
        #                     np.array([symbols[3]] * (Quantifier.scene_len - max_len))]))

        while True:
            random_counts = self._random_generate_constant_sum(len(symbols), Quantifier.scene_len)[0]

            # if this is not the required length regenerate the random numbers
            if min_len <= Quantifier.scene_len - random_counts[-1] <= max_len:
                continue

            # build the scene with the valid counts
            scene = np.random.permutation(dict(*zip(symbols, random_counts)))
            # sanity check that the scene we generated gives a quantifier true scene
            assert (self.quantify(scene)), "{} == False".format(self.name(scene="quantifier generated"))

    def generate_scenes(self, scene_num=scene_num, min_len=0, max_len=scene_len):
        """
        Generate scene_num scenes that gives a truth value under the quantifier,

        :param scene_num: number of scenes to generate
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: generated scenes
        """
        # return concatenated generated scenes to generate a matrix for training
        scenes = [self.generate_scene(min_len, max_len)
                  for _ in range(scene_num)]
        return np.vstack(scenes)

    @staticmethod
    def scene(counts, min_len=0, max_len=scene_len):
        """
        receives a count dictionary for each symbol and returns a random scene with
        ab count symbols marked 0 representing elements in A and B.
        a_b count symbols marked 1 representing elements in A but not in B.
        b_a count symbols marked 2 representing elements in B but not in A.
        c count symbols for the rest of the scene marked 3 meaning irrelevant (complement) elements
        first the symbols are put in order, generating a scene prototype
        then if required then the prototype is permuted to generate the final scene.

        :param counts: the input counts
        :param min_len: minimal number of scene symbols (apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (apart from the don't care symbols)

        :return: the constructed scene
        """
        # parameters sanity check
        assert (0 <= min_len <= max_len <= Quantifier.scene_len), "Illegal scene length limits given"
        assert (all([count >= 0 for count in counts.values()])), "Negative symbol count given"
        assert (sum(counts.values()) == Quantifier.scene_len), "Symbol counts don't add up to scene length"

        # generate and return the permuted scene prototype
        prototype = np.concatenate([[int(symbol)] * counts[symbol]
                                    for symbol in symbols]).astype(int)
        return prototype

    # scene quantification methods

    def quantify(self, scene):
        # evaluate the scene under the quantifier
        return True


class QuantityQuantifier(Quantifier):
    """
    base Natural Quantity compliant Quantifier class which implements generative and quantification methods
    by defining the inherited method constraints() which should return the constrains of each quantifier
    (constraints are defined over the scene counts implying implicitly that the quantity characteristic is maintained)
    by default constrains returns an empty list, inheriting classes need to return their specific constraints
    """

    # generative methods

    def generate_scene(self, min_len=0, max_len=Quantifier.scene_len):
        """
        generate a scene that gives a truth value under the quantifier, using the quantifier constraints

        :param min_len: minimal number of scene symbols (all symbols apart from the don't care symbols)
        :param max_len: maximal number of scene symbols (all symbols apart from the don't care symbols)

        :return: the generated scene (list of symbols)
        """
        assert (0 <= min_len <= max_len <= Quantifier.scene_len), "Illegal scene length limits given"

        counts = {}

        # go over the constraints twice and see all constant constraints don't contradict
        for _ in range(2):
            for constraint in self.constraints():
                if not constraint.comply(counts):
                    raise ValueError("Incompatible constraints can't generate scene")

        basic_len = sum(counts.values())
        assert (basic_len <= max_len), "quantifier has minimal symbols limit ({max_len})".format(max_len=max_len)

        # TODO: go over constraints pairs to look for contradictions (otherwise the following loop might be infinite)
        # loop till we get counts that comply with the constraints
        while True:
            random_counts = self._random_generate_constant_sum(len(symbols) - len(counts.keys()),
                                                               Quantifier.scene_len - basic_len)[0]

            # if this is not the required length regenerate the random numbers
            if not (min_len <= Quantifier.scene_len - random_counts[-1] <= max_len):
                continue

            # fill in the counts with the random generated counts
            filled_counts = counts.copy()
            i = 0
            for symbol in symbols:
                if symbol not in counts:
                    filled_counts[symbol] = random_counts[i]
                    i += 1

            # if scene counts comply with all quantifier constraints
            if self.quantification(filled_counts):
                # build the scene with the valid counts
                scene = np.random.permutation(self.scene(filled_counts))
                # sanity check that the scene we generated gives a quantifier true scene
                assert (self.quantify(scene)), "{} == False".format(self.name(scene="quantifier generated"))
                # return the generated scene
                return scene

    # scene quantification methods

    def quantify(self, scene):
        # evaluate the scene by calling the quantifier with the counts of the different symbols
        return self.quantification({symbol: defaultdict(int, zip(*np.unique(scene, return_counts=True)))[symbol]
                                    for symbol in symbols})

    def quantification(self, counts):
        # check that counts are compliant with all the constraints
        return all([constraint.comply(counts) for constraint in self.constraints()])

    def constraints(self):
        """
        returns the constraints that limit the quantifier's symbol counts
        by default the QuantityQuantifier base class is unconstrained and it returns an empty list
        """
        return []


class Most(QuantityQuantifier):
    def constraints(self):
        # most as are bs means that there are more as that are bs than the as that are not bs
        return [LinearRangeConstraint(symbol=ab_symbol, restriction=a_b_symbol)]


class Many(QuantityQuantifier):
    def constraints(self):
        # many as are bs means that there are more as that are bs than twice the as that are not bs
        return [LinearRangeConstraint(symbol=ab_symbol, restriction=a_b_symbol, alpha=2)]


class Few(QuantityQuantifier):
    def constraints(self):
        # few as are bs means that there are less as that are bs than twice the as that are not bs
        return [LinearRangeConstraint(symbol=a_b_symbol, restriction=ab_symbol, alpha=2)]


class Only(QuantityQuantifier):
    def constraints(self):
        # only as are bs means that there are no bs that are not as (non conservative quantifier)
        # this is a non conservative constraint as it applies to bs who are not as symbols
        return [ConstantConstraint(symbol=b_a_symbol, constant=0)]


class Some(QuantityQuantifier):
    def constraints(self):
        # some as are bs means that more as are bs than 0 (>=1)
        return [ConstantRangeConstraint(symbol=ab_symbol, restriction=1)]


class AFew(QuantityQuantifier):
    def constraints(self):
        # a few as are bs means that more as are bs than 1 (>=2)
        return [ConstantRangeConstraint(symbol=ab_symbol, restriction=2)]


class No(QuantityQuantifier):
    def constraints(self):
        # no as are bs means that no as are bs (==0)
        # we also add the constraint that some as are not bs so that we have some as in the context (>=1)
        return [ConstantConstraint(symbol=ab_symbol),
                ConstantRangeConstraint(symbol=a_b_symbol, restriction=1)]


class All(QuantityQuantifier):  # also called Every
    def constraints(self):
        # all as are bs means that no as are not bs (==0)
        # we also add the constraint that some as are bs so that we have some as in the context (>=1)
        return [ConstantConstraint(symbol=a_b_symbol),
                ConstantRangeConstraint(symbol=ab_symbol, restriction=1)]


class All2(QuantityQuantifier):  # also called Every
    def constraints(self):
        # all2 as are bs means that no as are not bs (==0)
        # we also add the constraint that some as are bs so that we have some as in the context (>=2)
        return [ConstantConstraint(symbol=a_b_symbol),
                ConstantRangeConstraint(symbol=ab_symbol, restriction=2)]


class The(QuantityQuantifier):
    def constraints(self):
        # the a is b means that there is exactly one a and it is also a b (no as that are not bs)
        return [ConstantConstraint(symbol=ab_symbol, constant=1),
                ConstantConstraint(symbol=a_b_symbol)]


class Both(QuantityQuantifier):
    def constraints(self):
        # both as are bs means that there are exactly two as and they are both also bs (no as that are not bs)
        return [ConstantConstraint(symbol=ab_symbol, constant=2),
                ConstantConstraint(symbol=a_b_symbol)]


class N(QuantityQuantifier):
    def __init__(self, n):
        self._n = n

    def constraints(self):
        # n as are bs means that there are at least n as that are bs
        return [ConstantRangeConstraint(symbol=ab_symbol, restriction=self._n)]

# compositional quantifier classes


class Conjunction(QuantityQuantifier, metaclass=ABCMeta):
    def __init__(self, quantifiers):
        self._quantifiers = quantifiers

    def name(self):
        return "{name}({quantifiers})".format(
            name=self.__class__.__name__,
            quantifiers=','.join([quantifier.name() for quantifier in self._quantifiers]))


class Or(Conjunction):
    def quantification(self, counts):
        return any([quantifier.quantification(counts) for quantifier in self._quantifiers])

    def constraints(self):
        return np.random.choice(self._quantifiers).constraints()


class And(Conjunction):
    def quantification(self, counts):
        return all([quantifier.quantification(counts) for quantifier in self._quantifiers])

    def constraints(self):
        return [constraint for quantifier in self._quantifiers for constraint in quantifier.constraints()]


class Operator(QuantityQuantifier, metaclass=ABCMeta):
    def __init__(self, quantifier):
        self._quantifier = quantifier

    def name(self):
        return "{name}({quantifier})".format(
            name=self.__class__.__name__,
            quantifier=self._quantifier.name())


class Not(Operator):
    def constraints(self):
        return [constraint.reversed() for constraint in self._quantifier.constraints()]

# non natural (non monotone) quantifiers


class ExactlyN(QuantityQuantifier):
    def __init__(self, n):
        self._n = n

    def constraints(self):
        # exactly n as are bs means that there are exactly n as that are bs
        return [ConstantConstraint(symbol=ab_symbol, constant=self._n)]


class Between(QuantityQuantifier):
    def __init__(self, min_a, max_a):
        assert (min_a < max_a)
        self._min_a = min_a
        self._max_a = max_a

    def constraints(self):
        return [ConstantRangeConstraint(symbol=ab_symbol, restriction=self._min_a),
                ConstantRangeConstraint(symbol=ab_symbol, restriction=self._max_a + 1, reverse=True)]


class Even(QuantityQuantifier):
    def constraints(self):
        return [EvenConstraint(symbol=ab_symbol)]


class Odd(QuantityQuantifier):
    def constraints(self):
        return [EvenConstraint(symbol=ab_symbol).reversed()]

# non natural quantity compliant quantifiers


class FirstN(Quantifier):
    def __init__(self, n):
        self._n = n

    def generate_scene(self, min_len=0, max_len=Quantifier.scene_len):
        while True:
            random_counts = self._random_generate_constant_sum(len(symbols), Quantifier.scene_len)[0]

            # if this is not the required length regenerate the random numbers
            if not (min_len <= Quantifier.scene_len - random_counts[-1] <= max_len):
                continue

            return np.array(self._random_interleave([ab_symbol] * (self._n + random_counts[0]) +
                                                    [a_b_symbol] * random_counts[1],
                                                    np.random.permutation([b_a_symbol] * random_counts[2] +
                                                                          [c_symbol] * random_counts[3]).tolist()))

    def quantify(self, scene):
        count = 0
        for symbol in scene:
            if symbol in [c_symbol, b_a_symbol]:
                continue
            if count < self._n:
                if symbol is ab_symbol:
                    count += 1
                else:
                    return False
            else:
                return True
        return True


class ExactlyFirstN(Quantifier):
    def __init__(self, n):
        self._n = n

    def generate_scene(self, min_len=0, max_len=Quantifier.scene_len):
        while True:
            random_counts = self._random_generate_constant_sum(len(symbols) - 1, Quantifier.scene_len - self._n)[0]

            # if this is not the required length regenerate the random numbers
            if not (min_len <= Quantifier.scene_len - random_counts[-1] <= max_len):
                continue

            return np.array(self._random_interleave([ab_symbol] * self._n + [a_b_symbol] * random_counts[0],
                                                    np.random.permutation([b_a_symbol] * random_counts[1] +
                                                                          [c_symbol] * random_counts[2]).tolist()))

    def quantify(self, scene):
        count = 0
        for symbol in scene:
            if symbol in [c_symbol, b_a_symbol]:
                continue
            if count < self._n:
                if symbol is ab_symbol:
                    count += 1
                else:
                    return False
            else:
                if symbol is ab_symbol:
                    return False
        return True
