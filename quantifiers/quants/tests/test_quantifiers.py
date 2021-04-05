import unittest

# my class imports

from quants.quantifiers import *

quantifiers = [No(), The(), Both(), Few(), Some(), AFew(), Many(), Most(), All(), Not(Most()), N(3), ExactlyN(5),
               Between(5, 150), Between(2, 50), Between(8, 40), Between(12, 35), FirstN(3), ExactlyFirstN(5)]

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
            # All quantifiers q entail not Not(q)
            print("Running entailment tests for scene_num={scene_num} min_len={min_len} max_len={max_len}..."
                  .format(scene_num=scene_num, min_len=min_len, max_len=max_len))
            # The entails Some
            assert(all([Some().quantify(scene) and not Not(The()).quantify(scene)
                        for scene in The().generate_scenes(scene_num, min_len, max_len)]))
            # Both entails AFew and Some
            assert(all([AFew().quantify(scene) and Some().quantify(scene) and not Not(Both()).quantify(scene)
                        for scene in Both().generate_scenes(scene_num, min_len, max_len)]))
            # AFew entails Some
            assert(all([Some().quantify(scene) and not Not(AFew()).quantify(scene)
                        for scene in AFew().generate_scenes(scene_num, min_len, max_len)]))
            # Few entails some
            assert(all([Some().quantify(scene) and not Not(Few()).quantify(scene)
                        for scene in Few().generate_scenes(scene_num, min_len, max_len)]))
            # Most entails some
            assert(all([Some().quantify(scene) and not Not(Most()).quantify(scene)
                        for scene in Most().generate_scenes(scene_num, min_len, max_len)]))
            # Many entails Most
            assert(all([Most().quantify(scene) and not Not(Many()).quantify(scene)
                        for scene in Many().generate_scenes(scene_num, min_len, max_len)]))
            # All entails Most and Some
            assert(all([Most().quantify(scene) and Some().quantify(scene) and not Not(All()).quantify(scene)
                        for scene in All().generate_scenes(scene_num, min_len, max_len)]))
            # Even entails not Odd
            assert(all([not Odd().quantify(scene) and not Not(Even()).quantify(scene)
                        for scene in Even().generate_scenes(scene_num, min_len, max_len)]))
            # Odd entails not Even
            assert(all([not Even().quantify(scene) and not Not(Odd()).quantify(scene)
                        for scene in Odd().generate_scenes(scene_num, min_len, max_len)]))
            # FirstN entails N
            for n in [3, 5, 7]:
                assert(all([N(n).quantify(scene)
                            for scene in FirstN(n).generate_scenes(scene_num, min_len, max_len)]))
            # ExactylFirstN entails ExactlyN
            for n in [3, 5, 7]:
                assert(all([N(n).quantify(scene)
                            for scene in ExactlyFirstN(n).generate_scenes(scene_num, min_len, max_len)]))
            # test Or and And conjunctions using Between (Between(n1,n2) entails (n1 and not n2)==(not(n2 or not n1)))
            for n1, n2 in zip([3, 5, 7], [7, 9, 20]):
                assert(all([And([N(n1), Not(N(n2 + 1))]).quantify(scene)
                            and
                            Not(Or([(N(n2 + 1)), Not(N(n1))])).quantify(scene)
                            for scene in Between(n1, n2).generate_scenes(scene_num, min_len, max_len)]))


if __name__ == '__main__':
    unittest.main()
