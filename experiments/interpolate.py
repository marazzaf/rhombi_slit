import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import LinearNDInterpolator

def BC():
    #Load data
    data = np.loadtxt('./experiments/non_pull_ref.txt', comments='#')

    #Creating the Linear interpolation
    interp = LinearNDInterpolator(data[:,2:], data[:,2])
    return interp


















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































