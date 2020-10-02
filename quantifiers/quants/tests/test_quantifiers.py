# Quantifier unit test

# imports

import unittest

# my class imports

from quants.quantifiers import *

# TODO: generate quantifiers usign inspection of Quantifier Heirarchy
quantifiers = [The(), Both(), No(), All(), Some(), Most(),
               MinMax(2, 5), MinMax(8, 10), MinMax(12, 15), MinMax(17, 20), MinMax(24, 30), MinMax(37, 50)]


class TestQuantifiers(unittest.TestCase):
    def test_generation(self):
        """ generate quantifiers (this internally checks that the generated scenes are quantifier True) """
        [quantifier.generate_scenes() for quantifier in quantifiers]
        [quantifier.generate_scenes(max_len=400) for quantifier in quantifiers]

    def test_entailment(self):
        """ test that all quantifiers are True for quantifiers they entail """
        # The(), Both(), No(), All(), Some(), Most()
        # The entails Some
        assert(all([Some().quantify(scene) for scene in The().generate_scenes()]))
        # Both entails Some
        assert(all([Some().quantify(scene) for scene in Both().generate_scenes()]))
        # Few entails Some
        assert(all([Some().quantify(scene) for scene in Few().generate_scenes()]))
        # Most entails Some
        assert(all([Some().quantify(scene) for scene in Most().generate_scenes()]))
        # All entails Most and Some
        assert(all([Most().quantify(scene) and Some().quantify(scene) for scene in All().generate_scenes()]))


if __name__ == '__main__':
    unittest.main()
