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
np.seterr(all="ignore")
import pandas as pd
# Plotting library
from matplotlib import pyplot as plt
# Optimization module in scipy
from scipy import optimize
# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Show Debug info
DEBUG = False

def load_clean_normalize_balance_split_and_save_dataset():
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

    utils.showCorrelation(df)
    if DEBUG:
        utils.showProportion(df, "Unbalanced Class Proportion")

    print("Normalizing dataset")
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
    if DEBUG:
        utils.showProportion(clean_df, "Balanced Class Proportion")

    '''
    Split dataset into 70/15/15 train, validate, test
    '''
    # Split dataset 
    train, validate, test = np.split(clean_df.sample(frac=1, random_state=42),[int(.7*len(clean_df)), int(.85*len(clean_df))])
    train.to_csv("train.csv")
    validate.to_csv("validate.csv")
    test.to_csv("test.csv")
    
train=pd.read_csv('train.csv')
validate=pd.read_csv('validate.csv')
test=pd.read_csv('test.csv')

'''
Train model
'''
# Generate lambdas
bigLambdas = list(range(0,100))
lambdas = []
for l in bigLambdas:
    lambdas.append(l/250)

models  = []

X_train = pd.DataFrame(data=train, columns=['u', 'g', 'r', 'i', 'z', 'redshift']).to_numpy(dtype=float)
y_train = train['class'].to_numpy()

for lambda_ in lambdas:
    print("--------------------\n   Training model   \n--------------------\nMethod: One Vs All\nLambda: ",lambda_)
    model = utils.oneVsAll(X_train, y_train, 3, lambda_)
    #print("Thetas: ", model)

    # Validate model using validation data and determine accuracy
    accuracy = utils.validateModel(validate, model)

    # Append this model to cache
    modelResult = {
        "Lambda": lambda_, 
        "Accuracy": accuracy, 
        "Model": model
    }
    models.append(modelResult)
    print("Accuracy: ", accuracy)

# Show performace in function of lambda
accuracy = []
for curModel in models:
    accuracy.append(curModel["Accuracy"])
plt.plot(lambdas, accuracy)
plt.xlabel("Lambda")
plt.ylabel("Performance (%)")
plt.show()

# Determine best performing model (Best Hyperparameter value)
bestModel = {
    "Lambda" : 0,
    "Accuracy" :0,
    "Model": None
}

for curModel in models:
    print("Lambda: ",curModel['Lambda'], " \t- Accuracy: ",curModel['Accuracy'])
    if (curModel["Accuracy"] > bestModel["Accuracy"]):
        # Replace with better model
        bestModel = curModel

print("Best model has an accuracy of ", bestModel["Accuracy"])

'''
Predict dataset
'''
print("Predicting dataset with test dataset...")

X_t = pd.DataFrame(data=test, columns=['u', 'g', 'r', 'i', 'z', 'redshift']).to_numpy(dtype=float)
y_t = test['class'].to_numpy()

pred = utils.predictOneVsAll(bestModel["Model"], X_t)
accuracy = np.mean(pred == y_t) * 100

print("--------------------\nTraining Set Accuracy\n--------------------\n",accuracy,"%")

# Report as graph
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, hamming_loss

y_t = np.array(y_t, dtype=int)
pred = np.array(pred, dtype=int)
print("Raw result: ", y_t, pred)
accuracy, confusion, hamming = accuracy_score(y_t, pred), confusion_matrix(y_t, pred), hamming_loss(y_t, pred)
print("Accuracy Score: ", accuracy)
print("Confusion Matrix: ", confusion)
print("Hamming Loss: ", hamming)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j, y=i,s=confusion[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
