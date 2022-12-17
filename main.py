'''
Kaggle Dataset - Stellar Classification Dataset - SDSS17
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
https://www.kaggle.com/code/psycon/stars-galaxies-eda-and-classification

Made by Massimo Giardina and Ismael Warnants
--------------
One vs All Classification Logistic Regression 
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
X = pd.DataFrame(data=df, columns=['u', 'g', 'r', 'i', 'z', 'redshift']) #.to_numpy()
y = df['class']#.to_numpy()

# Normalize data per column
X = (X-X.min())/(X.max()-X.min())

# Show dataset
if DEBUG:
    print("X_norm shape: ",X.shape)

    for col in X:
        print(col)
    X.plot(kind='box')

    plt.show()
df = X
'''

'''
Balance dataset
'''
print("Balancing dataset")

twos_subset = df.loc[df["class"] == 2, :]
number_of_twos = twos_subset.shape[0]
print("Number of 2s: ", number_of_twos)

zeros_subset = df.loc[df["class"] == 0, :]
sampled_zeros = zeros_subset.sample(number_of_twos)
number_of_zeros = sampled_zeros.shape[0]
print("Number of 0s: ", number_of_zeros)

ones_subset = df.loc[df["class"] == 1, :]
sampled_ones = ones_subset.sample(number_of_twos)
number_of_ones = sampled_ones.shape[0]
print("Number of 1s: ", number_of_ones)

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

#train, validate, test = np.split(clean_df.sample(frac=1, random_state=42),[int(.7*len(clean_df)), int(.85*len(clean_df))])

'''
Train model
'''
print("Training model...")
X = pd.DataFrame(data=train, columns=['u', 'g', 'r', 'i', 'z', 'redshift']).to_numpy(dtype=float)
y = train['class'].to_numpy()
lambda_ = 0.01

all_theta = utils.oneVsAll(X, y, 3, lambda_)
print("--------------------\nOneVsAll Thetas\n--------------------\n",all_theta)

'''
X = train.iloc[:,0:4]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y = train.iloc[:,5].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,5])
print("Theta shape: ",theta.shape)

# Apply one-vs-all multi-classification
lambda_ = 0.1
print("X Shape:", X.shape, "Y Shape:", y.shape)
all_theta = utils.oneVsAll(X, y, 3, lambda_)

print("Thetas:\n",all_theta)
'''

'''
Predict dataset
'''
print("Predicting dataset...")
'''
print(test.head())
X_test = validate.iloc[:,0:5]
y_test = validate.iloc[:,5:10].values
print(X_test.shape)
print(y_test.shape)

pred = utils.predictOneVsAll(all_theta, X_test)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y_test) * 100))
'''
X_t = pd.DataFrame(data=test, columns=['u', 'g', 'r', 'i', 'z', 'redshift']).to_numpy(dtype=float)
y_t = test['class'].to_numpy()

pred = utils.predictOneVsAll(all_theta, X_t)
accuracy = np.mean(pred == y_t) * 100

print("--------------------\nTraining Set Accuracy\n--------------------\n",accuracy,"%")
