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
DEBUG = True

# Load dataset 
df=pd.read_csv('star_classification.csv')

print('Shape before filtering columns :',df.shape)
# Change class to values
df.loc[df["class"] == "GALAXY", "class"]    = 0
df.loc[df["class"] == "STAR", "class"]      = 1
df.loc[df["class"] == "QSO", "class"]       = 2

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
df = (df - df.mean())/df.std()
df['class'] = dfclass.copy()

# Show dataset
print("DATAFRAME COLUMNNS: ",df.columns)
X = pd.DataFrame(data=df, columns=['u', 'g', 'r', 'i', 'z', 'redshift']) #.to_numpy()
y = df['class']#.to_numpy()

# Normalize data per column
X = (X-X.min())/(X.max()-X.min())

if DEBUG:
    fig = plt.figure()
    print("X_norm shape: ",X.shape)

    for col in X:
        print(col)
    X.plot(kind='box')

    plt.show()

print("DataFrame Head: \n",df.head())

df = X

# Balancing dataset
twos_subset = df.loc[df["class"] == 2, :] # Lowest present
number_of_2s = len(twos_subset)

print(number_of_2s)

zeros_subset = df.loc[df["class"] == 0, :]
sampled_zeros = zeros_subset.sample(number_of_2s) # Sampling the same amount as the lowest present

print(sampled_zeros)


ones_subset = df.loc[df["class"] == 1, :]
sampled_ones = ones_subset.sample(number_of_2s) # Sampling the same amount as the lowest present

print(sampled_ones)

clean_df = pd.concat([sampled_ones, sampled_zeros, twos_subset], ignore_index=True)

print(clean_df)

print(clean_df['class'].value_counts())


# Split dataset 
train, validate, test = np.split(clean_df.sample(frac=1, random_state=42),[int(.7*len(clean_df)), int(.85*len(clean_df))])

print(train['class'].value_counts())
print(validate['class'].value_counts())
print(test['class'].value_counts())

X = train.iloc[:,0:4]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = train.iloc[:,5].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,5])
print(theta.shape)

# Apply first layer
# Train hidden layer
# Test model