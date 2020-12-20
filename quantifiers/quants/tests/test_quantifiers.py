import unittest

# my class imports

from quants.quantifiers import *

# TODO: test conjunction quantifiers

quantifiers = [No(), The(), Both(), Few(), Some(), AFew(), Many(), Most(), All(),
               Between(2, 50), Between(8, 40), Between(12, 35)]

test_case_parameters = ([1000, 200, 150, 100], [0, 100, 300, 0], [500, 400, 500, 200])


class TestQuantifiers(unittest.TestCase):
    def test_short_display(self):
        for quantifier in quantifiers:
            symbol_counts = defaultdict(int, zip(*np.unique(quantifier.generate_scene(), return_counts=True)))
            symbol_counts = {symbols[symbol]: symbol_counts[symbol] for symbol in range(len(symbols))}
            print(quantifier.name(), symbol_counts)

    def test_generation(self):
        """ generate quantifiers (this internally checks that the generated scenes are quantifier True) """
        # do this for various scene length limits and overall scene length
        for scene_num, min_len, max_len in zip(*test_case_parameters):
            for quantifier in quantifiers:
                quantifier.generate_scenes(scene_num, min_len, max_len)

    def test_entailment(self):
        """ test all quantifier generated scenes are True for quantifiers that their generating quantifiers entail """
        # No(), The(), Both(), Some(), AFew(), Few(), Many(), Most(), All()

        # do this for various scene length limits
        for scene_num, min_len, max_len in zip(*test_case_parameters):
            # The entails Some
            assert(all([Some().quantify(scene)
                        for scene in The().generate_scenes(scene_num, min_len, max_len)]))
            # Both entails Some
            assert(all([Some().quantify(scene)
                        for scene in Both().generate_scenes(scene_num, min_len, max_len)]))
            # AFew entails Some
            assert(all([Some().quantify(scene)
                        for scene in AFew().generate_scenes(scene_num, min_len, max_len)]))
            # Most entails Some
            assert(all([Some().quantify(scene)
                        for scene in Most().generate_scenes(scene_num, min_len, max_len)]))
            # All entails Most and Some
            assert(all([Most().quantify(scene) and Some().quantify(scene)
                        for scene in All().generate_scenes(scene_num, min_len, max_len)]))
            # Both entails AFew and Some
            assert(all([AFew().quantify(scene) and Some().quantify(scene)
                        for scene in Both().generate_scenes(scene_num, min_len, max_len)]))


if __name__ == '__main__':
    unittest.main()
