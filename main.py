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

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Dataset 
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

print("The input data shape is ", X.shape, "; there are ", X[:,1].size, " training examples and ", X[1,:].size, " features.")

# 3 labels (Galaxy/GALAXY, Star/STAR or Quasar/QSO)
num_labels = 3
# -> y = [100000; 1] vector that contain labels for training set.

'''
    Convert string labels to classification values
    GALAXY = 0
    STAR = 1
    QUASAR = 2
'''
y = np.ones(labels.size)
for i in range(y.size):
    if (labels[i] == "GALAXY"):
        y[i] = 0
    elif (labels[i] == "STAR"):
        y[i] = 1
    elif (labels[i] == "QSO"):
        y[i] = 2
    else:
        print("ERROR: ",i,labels[i],"Unknown classifier in dataset!")

# Apply one-vs-all multi-classification
lambda_ = 0.1
print(X)
print(X.shape)
print(y)
print(y.shape)
all_theta = utils.oneVsAll(X, y, num_labels, lambda_)

# Predict dataset
#pred = utils.predictOneVsAll(all_theta, X)
#print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

