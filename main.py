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

import pandas as pd

# Dataset 
#labels = np.genfromtxt("MachineLearning_2223/star_classification.csv", delimiter=",", skip_header=1, usecols=17, dtype=str) # use arg names=True for headers

#loading the data
df=pd.read_csv('MachineLearning_2223/star_classification.csv')
print(df['class'].value_counts())
#plt.pie(df['class'].value_counts(),autopct="%1.1f%%",labels=['GALAXY','STAR','QSO'])
#plt.legend()
#plt.show()

#   The heatmap on raw data
#sns.heatmap(df.corr(numeric_only=True))
#plt.show()

print('Shape before filtering columns :',df.shape)

# Change class to values
df.loc[df["class"] == "GALAXY", "class"] = 0
df.loc[df["class"] == "STAR", "class"] = 1
df.loc[df["class"] == "QSO", "class"] = 2
print(df['class'].value_counts())

#dfclass['class'] = df['class'].copy()

df.drop(['obj_ID','cam_col', 'run_ID', 'rerun_ID', 'field_ID', 'fiber_ID', 'plate', 'MJD', 'spec_obj_ID', 'alpha', 'delta', 'redshift'] ,axis=1, inplace=True)
df.shape

print('Shape before filtering outliers :',df.shape)
df=df[df.z>-2000]
df=df[df.u>-2000]
df=df[df.g>-2000]
print('Shape after filtering :',df.shape)

#   The heatmap looks way better now
#sns.heatmap(df.corr(numeric_only=True))
#plt.show()

dfclass = df['class'].copy()
df = (df - df.mean())/df.std()
df['class'] = dfclass.copy()
#print(df.head())
#print(dfclass)

print(df.head())

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

# Apply one-vs-all multi-classification
lambda_ = 0.1
print("X Shape:", X.shape, "Y Shape:", y.shape)
all_theta = utils.oneVsAll(X, y, 3, lambda_)

print(all_theta)

# Predict dataset
print(test.head())
X_test = validate.iloc[:,0:5]
y_test = validate.iloc[:,5:10].values
print(X_test.shape)
print(y_test.shape)
pred = utils.predictOneVsAll(all_theta, X_test)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y_test) * 100))

