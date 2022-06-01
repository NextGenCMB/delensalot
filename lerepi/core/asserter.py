"""Decorators to check configurations
"""

def lerepi_asserter(func):
    def func_wrapper(*args):
        assert args[0].K in ['p_p', 'p_eb', 'ptt'], 'Not supported'
        """Put things here which you'd like to happen
        """
        return func(*args)
    return func_wrapper


def dlensalot_asserter(func):
    def func_wrapper(*args):
        """Put things here which you'd like to happen
        """
        return func(*args)
    return func_wrapper