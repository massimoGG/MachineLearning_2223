'''
Kaggle Dataset - Stellar Classification Dataset - SDSS17
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
https://www.kaggle.com/code/psycon/stars-galaxies-eda-and-classification

Made by Massimo Giardina and Ismael Warnants
'''

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Dataset 
# 3 labels (Galaxy, Star or Quasar)
num_labels = 3

#  training data stored in arrays X, y
data = np.genfromtxt("star_classification.csv", delimiter=",", dtype=str)
#loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data[:,:17], data[:,17]
#X, y = data['X'], data['y'].ravel()

m = y.size

print("Dataset size: ", m)
print("Dataset shape: ", data.shape)
print("input shape: ", X.shape)
print("Labels: ", y)
