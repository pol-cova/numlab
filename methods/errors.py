import numpy as np
import math

def get_relative_error(real, approx):
    '''
    This function calculates the relative error of a given real value and an approximate value.
    '''
    return abs(real - approx) / abs(real)

def get_absolute_error(real, approx):
    '''
    This function calculates the absolute error of a given real value and an approximate value.
    '''
    return abs(real - approx)

def get_real_error(real, approx):
    '''
    This function calculates the real error of a given real value and an approximate value.
    '''
    return real - approx