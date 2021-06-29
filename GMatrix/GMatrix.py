from itertools import chain
import word2vec

model = word2vec.getModel()

hebrew = 'אבגדהוזחטיכלמנסעפצקרשתךםןףץ'

GMatrix = {"normal": {c:v for c, v in zip(hebrew, 
                                          chain(range(1, 10),
                                                range(10, 100, 10),
                                                range(100, 500, 100),
                                                [20, 40, 50, 80, 90])
                                         )},
           "big": {c:v for c, v in zip(hebrew, 
                                       chain(range(1, 10),
                                             range(10, 100, 10),
                                             range(100, 1000, 100))
                                         )},
           "small": {c:v for c, v in zip(hebrew, 
                                         chain(range(1, 10),
                                               range(1, 10),
                                               range(1, 10)))},
           "serial": {c:v for c, v in zip(hebrew, 
                                          chain(range(1, 28)))},
           "etbesh": {c:v for c, v in zip(hebrew, 
                                          chain(reversed(range(100, 500, 100)),
                                                reversed(range(10, 100, 10)),
                                                reversed(range(1, 10)),
                                                [30, 10, 9, 7, 6])
                                         )},
          }

GMatrix["forward"] = {c:v for c, v in zip(hebrew, np.cumsum(GMatrix["normal"]))}
GMatrix["forward"][:-5] = GMatrix["forward"][11, 13, 14, 17, 18]
GMatrix["private"] = {c:v for c, v in zip(hebrew, map(lambda x: x ** 2, GMatrix["normal"]))}
GMatrix["semite"] = {c:v for c, v in zip(hebrew, [GMatria('אלף'),
                                                  GMatria('בית'),
                                                  GMatria('גמל'),
                                                  GMatria('דלת'),
                                                  GMatria('הא'),
                                                  GMatria('ויו'),
                                                  GMatria('זיין'),
                                                  GMatria('חית'),
                                                  GMatria('טית'),
                                                  GMatria('יוד'),
                                                  GMatria('כף'),
                                                  GMatria('למד'),
                                                  GMatria('מם'),
                                                  GMatria('נון'),
                                                  GMatria('סמך'),
                                                  GMatria('עין'),
                                                  GMatria('פא'),
                                                  GMatria('צדי'),
                                                  GMatria('קוף'),
                                                  GMatria('ריש'),
                                                  GMatria('תיו'),
                                                  GMatria('כף'),
                                                  GMatria('מם'),
                                                  GMatria('נון'),
                                                  GMatria('פא'),
                                                  GMatria('צדי')]

def hebrew_str(s):
    return all("\u0590" <= c <= "\u05EA" or c in " ,.:;'[](){}" for c in s)

def GMatria(s, method="normal"):
    if not hebrew_str(s):
        pass
#         print('None Hebrew letter encountered')
    return(sum(GMatrix[method][c] for c in s if c in GMatrix[method]))


from collections import namedtuple
# This is a doubly linked list.
# (value, tail) will be one group of solutions.  (next_answer) is another.
SumPath = namedtuple('SumPath', 'value tail next_answer')

def fixed_sum_paths (array, target, count):
    # First find counts of values to handle duplications.
    value_repeats = {}
    for value in array:
        if value in value_repeats:
            value_repeats[value] += 1
        else:
            value_repeats[value] = 1

    # paths[depth][x] will be all subsets of size depth that sum to x.
    paths = [{} for i in range(count+1)]

    # First we add the empty set.
    paths[0][0] = SumPath(value=None, tail=None, next_answer=None)

    # Now we start adding values to it.
    for value, repeats in value_repeats.items():
        # Reversed depth avoids seeing paths we will find using this value.
        for depth in reversed(range(len(paths))):
            for result, path in paths[depth].items():
                for i in range(1, repeats+1):
                    if count < i + depth:
                        # Do not fill in too deep.
                        break
                    result += value
                    if result in paths[depth+i]:
                        path = SumPath(
                            value=value,
                            tail=path,
                            next_answer=paths[depth+i][result]
                            )
                    else:
                        path = SumPath(
                            value=value,
                            tail=path,
                            next_answer=None
                            )
                    paths[depth+i][result] = path

                    # Subtle bug fix, a path for value, value
                    # should not lead to value, other_value because
                    # we already inserted that first.
                    path = SumPath(
                        value=value,
                        tail=path.tail,
                        next_answer=None
                        )
    return paths[count][target]

def path_iter(paths):
    if paths.value is None:
        # We are the tail
        yield []
    else:
        while paths is not None:
            value = paths.value
            for answer in path_iter(paths.tail):
                answer.append(value)
                yield answer
            paths = paths.next_answer

def fixed_sums (array, target, count):
    paths = fixed_sum_paths(array, target, count)
    return path_iter(paths)

for path in fixed_sums([1,2,3,3,4,5,6,9], 10, 3):
    print(path)