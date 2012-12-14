""" Bunch class to replace the scikit one
"""
from collections import namedtuple

def Bunch(**kwargs):
    return namedtuple('Bunch', kwargs.keys())(**kwargs)
