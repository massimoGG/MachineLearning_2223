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
data = np.genfromtxt('star_classification.csv', delimiter=',', skip_header=1)[:,[3,4,5,6,7,13]]
'''
[:,:17] to stop before the string labels
[1:,..] to remove the header column 
'''

# 3 labels (Galaxy/GALAXY, Star/STAR or Quasar/QSO)
num_labels = 3

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

### Preprocess dataset ###
# Ignore useless columns/features
'''
Ik raad aan om de volgende kolommen te negeren (en dus verwijderen) aangezien ze geen nuttige data zijn:
- obj_ID            = A0
- Alpha             = B1
- Delta             = C2
- run_ID            = I8
- rereun_ID         = J9
- cam_col           = K10
- field_ID          = L11
- spec_obj_ID ?     = M12
- plate ?           = O14
- MJD               = P15
- FIBERID           = Q16
'''
X = data

fig, ax = pyplot.subplots(3)

# Show data [column based]
for i in range(0, X.shape[1]):
    col = X[:,i]
    Y_ = np.ones(col.size)*i
    ax[0].plot(Y_, col, 'ro', ms=2, mec='k')
    ax[0].set_title("RAW dataset")

# Filter data [row based]
i = 0
newX = np.ones(1)
for i in range(0, X.shape[0]):
    row = X[i,:]
    for j in row:
        if (j < -1):
            # Corrupted datapoint, remove
            newX = np.delete(X, i,0)
            y = np.delete(y,i,0)
            print("DELETED", i)
            break
    i += 1
X = newX
print("New dataset: ", X.shape)

# Show dataset
for i in range(0, X.shape[1]):
    col = X[:,i]
    Y_ = np.ones(col.size)*i
    # Filter out the -9999 points
    ax[1].plot(Y_, col, 'bo', ms=2, mec='k')
    ax[1].set_title("Filtered dataset")

# Normalize data per column
for i in range(0, X.shape[1]):
    # Scale between 0 and 1
    # Figure out min and max

    minValue = np.min(X[:,i])
    maxValue = np.max(X[:,i])
    print(i,"minValue",minValue,"maxValue",maxValue)
    X[:,i] = (X[:,i] - minValue) / (maxValue - minValue)

# Show mapped dataset
for i in range(0, X.shape[1]):
    col = X[:,i]
    Y_ = np.ones(col.size)*i
    # Filter out the -9999 points
    ax[2].plot(Y_, col, 'go', ms=2, mec='k')
    ax[2].set_title("Mapped dataset")

pyplot.show()

# Apply one-vs-all multi-classification
lambda_ = 0.1
print("X Shape:", X.shape, "Y Shape:", y.shape)
all_theta = utils.oneVsAll(X, y, num_labels, lambda_)

print(all_theta)

# Predict dataset
pred = utils.predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

