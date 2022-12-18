'''
Kaggle Dataset - Stellar Classification Dataset - SDSS17
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
https://www.kaggle.com/code/psycon/stars-galaxies-eda-and-classification

Made by Massimo Giardina and Ismael Warnants
--------------
Neural Network implementation

'''

# used for manipulating directory paths
import os
# Scientific and vector computation for python
import numpy as np
import pandas as pd
# Plotting library
from matplotlib import pyplot as plt
# Optimization module in scipy
from scipy import optimize
# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Show Debug info
DEBUG = False

# Load dataset 
#labels = np.genfromtxt("MachineLearning_2223/star_classification.csv", delimiter=",", skip_header=1, usecols=17, dtype=str) # use arg names=True for headers
df=pd.read_csv('star_classification.csv')
#plt.pie(df['class'].value_counts(),autopct="%1.1f%%",labels=['GALAXY','STAR','QSO'])
#plt.legend()
#plt.show()

print('Shape before filtering columns :',df.shape)
# Change class to values
df.loc[df["class"] == "GALAXY", "class"] = 0
df.loc[df["class"] == "STAR", "class"] = 1
df.loc[df["class"] == "QSO", "class"] = 2

# Drop useless fields
df.drop(['obj_ID','cam_col', 'run_ID', 'rerun_ID', 'field_ID', 'fiber_ID', 'plate', 'MJD', 'spec_obj_ID','alpha', 'delta'] ,axis=1, inplace=True)

print('Shape before filtering outliers :',df.shape)
df=df[df.z>-2000]
df=df[df.u>-2000]
df=df[df.g>-2000]
print('Shape after filtering :',df.shape)

if DEBUG:
    utils.showCorrelation(df)

dfclass = df['class'].copy()
#df = (df - df.mean())/df.std()
df = (df - df.min())/(df.max() - df.min())
df['class'] = dfclass.copy()

'''
Balance dataset
'''
print("Balancing dataset")

twos_subset = df.loc[df["class"] == 2, :]
number_of_twos = twos_subset.shape[0]

zeros_subset = df.loc[df["class"] == 0, :]
sampled_zeros = zeros_subset.sample(number_of_twos)
number_of_zeros = sampled_zeros.shape[0]

ones_subset = df.loc[df["class"] == 1, :]
sampled_ones = ones_subset.sample(number_of_twos)
number_of_ones = sampled_ones.shape[0]

clean_df = pd.concat([sampled_ones, sampled_zeros, twos_subset], ignore_index=True)

print("Balanced dataset: ")
print(clean_df)
print(clean_df['class'].value_counts())

'''
Split dataset into 70/15/15 train, validate, test
'''
train_size      = int(0.70*(clean_df.shape[0]))
validate_size   = int(0.15*(clean_df.shape[0]))
test_size       = int(0.15*(clean_df.shape[0]))

train           = clean_df.sample(train_size)
validate        = clean_df.sample(validate_size)
test            = clean_df.sample(test_size)

'''
Neural Network Layers
'''
input_layer_size  = clean_df.columns.size
hidden_layer_size = 25
num_labels        = 3

weights = None

# get the model weights from the dictionary
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# Feedfordward Propagation and Prediction