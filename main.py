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

# TODO VANAF HIER MOET ALLES NOG AANGEPAST WORDEN OP ONZE DATASET

# TODO: Test dataset toevoegen (Ik zou 80/20 nemen hiervoor?)
# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

# Convert string labels to classification values
y = labels

m = y.size

# Calculate cost function
J, grad = utils.lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))

# Apply one-vs-all multi-classification
lambda_ = 0.1
all_theta = utils.oneVsAll(X, y, num_labels, lambda_)

# Predict dataset
pred = utils.predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))