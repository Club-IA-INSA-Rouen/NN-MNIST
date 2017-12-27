import numpy as np

class CrossEntropyCost(object):
    """defines the quadratic cost function and associated delta value,
    where delta is the error of the last layer - full static"""
    @staticmethod
    def f(aL,y): # aL activation of last layer = a[-1]
        # be careful of NaN from log(a) & log(1-a) !
        return np.sum(np.nan_to_num(-y*np.log(aL)-(1-y)*np.log(1-aL)))

    @staticmethod
    def delta(z,aL,y): # where last layer aL=activation_fun(z) approximates y
        return (aL-y)

class QuadraticCost(object):
    """defines the quadratic cost function and associated delta value,
    where delta is the error of the last layer - full static"""
    @staticmethod
    def f(aL,y): # aL activation of last layer = a[-1]
        return 0.5*np.linalg.norm(aL,y)**2

    @staticmethod
    def delta(z,aL,y): # where last layer aL=activation_fun(z) approximates y
        return (aL-y)*aL*(1-aL)
