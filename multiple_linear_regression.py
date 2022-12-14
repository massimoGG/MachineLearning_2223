import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils

#loading the data
df=pd.read_csv('MachineLearning_2223/star_classification.csv')
print(df['class'].value_counts())
#plt.pie(df['class'].value_counts(),autopct="%1.1f%%",labels=['GALAXY','STAR','QSO'])
#plt.legend()
#plt.show()

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


#setting the matrixes
parameters = 10000
X = train.iloc[:parameters,0:4]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = train.iloc[:parameters,5:10].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,5])
print(theta.shape)

#set hyper parameters
alpha = 0.00000001
iters = 1000

#computecost
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

#running the gd and cost function
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)

#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
plt.show()