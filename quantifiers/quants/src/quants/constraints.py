# scene symbols
ab_symbol = 0  # elements in A and in B (A&B)
a_b_symbol = 1  # elements in A but not in B (A-B)
b_a_symbol = 2  # elements in B but not in A (B-A)
c_symbol = 3  # irrelevant (don't care) padding elements

symbols = [ab_symbol, a_b_symbol, b_a_symbol, c_symbol]


class Constraint:
    """
    A class that embodies constraints on the symbol counts in a scene
    this lies at the heart of both:
        1. generation of scenes that are True under the quantifier
        2. calculating quantifier Truth values
    """

    def __init__(self, symbol, reverse=False):
        """
        :param symbol: restricted symbol
        :param reverse: reverses the constraint
        """
        self._symbol = symbol
        self._reverse = reverse

    def reversed(self):
        """
        reverses the constraint
        :return: reversed version of this constraint
        """
        return self.__class__(self, reverse=not self._reverse)

    def comply(self, counts):
        """
        Check if the counts comply to the constraint, also sets count if constant
        :param counts: symbol counts
        :return: Truth value for the constraint given the counts
        """
        pass


class LinearRangeConstraint(Constraint):
    """
    Linear range constraint
    the symbol count is smaller than the restriction symbol times a constant alpha
    symbol >= alpha * restriction
    """

    def __init__(self, restriction, alpha=1, *argv, **kwargs):
        """
        :param restriction: symbol that restricts our constrained symbol counts
        :param alpha: linear multiplier
        """
        self._restriction = restriction
        self._alpha = alpha
        super().__init__(*argv, **kwargs)

    def comply(self, counts):
        if self._symbol in counts and self._restriction in counts:
            return self._reverse != (counts[self._symbol] >=
                                     self._alpha * counts[self._restriction])
        return True


class ConstantRangeConstraint(Constraint):
    """
    Constant value range constraint
    the symbol count is smaller than a constant restriction
    symbol >= constant
    """

    def __init__(self, restriction, *argv, **kwargs):
        """
        :param restriction: constant restriction on the constrained symbol counts
        """
        self._restriction = restriction
        super().__init__(*argv, **kwargs)

    def comply(self, counts):
        if self._symbol in counts:
            return self._reverse != (counts[self._symbol] >= self._restriction)
        return True


class ConstantConstraint(Constraint):
    """ Constant value constraint on a single symbol count """

    def __init__(self, constant=0, *argv, **kwargs):
        """
        :param constant: constant value for the constrained symbol counts
        """
        self._constant = constant
        super().__init__(*argv, **kwargs)

    def comply(self, counts):
        if self._symbol not in counts:
            counts[self._symbol] = self._constant
            return True
        return self._reverse != (counts[self._symbol] == self._constant)


class EvenConstraint(Constraint):
    """ Even constraint on a single symbol count """

    def comply(self, counts):
        return self._reverse != (counts[self._symbol] % 2 == 0)
