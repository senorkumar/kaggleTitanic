import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp
#import ipython
import sklearn
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import seaborn as sns
import pylab

#matplotlib inline
plt.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


data_raw = pd.read_csv("train.csv")
data_val = pd.read_csv("test.csv")

#working through https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy to learn more

data1 = data_raw.copy(deep = True)


data_cleaner = [data1, data_val]

print(data1.info())

print(data1.sample(10))

#iterate through train and test
for dataset in data_cleaner:
    #fill in missing data with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #fill in embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #fill in missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    #delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())
