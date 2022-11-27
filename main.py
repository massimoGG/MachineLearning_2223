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

labels = np.genfromtxt("star_classification.csv", delimiter=",", skip_header=1, usecols=17, dtype=str) # use arg names=True for headers
data = np.genfromtxt('star_classification.csv', delimiter=',', skip_header=1)[:,:17]
'''
[:,:17] to stop before the string labels
[1:,..] to remove the header column 
'''

### Preprocess dataset ###
# Ignore useless columns/features
'''
Ik raad aan om de volgende kolommen te negeren (en dus verwijderen) aangezien ze geen nuttige data zijn:
- obj_ID
- Alpha
- Delta
- run_ID
- rereun_ID
- cam_col
- field_ID
- spec_obj_ID ?
- plate ?
- MJD
'''

X = data 

# Convert string labels to classification values
y = labels

m = y.size

print("Dataset size: ", m)
print("Dataset shape: ", data.shape)
print("input shape: ", X.shape)
print("Labels: ", y)
print(X[1])


