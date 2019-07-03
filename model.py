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

#done cleaning data, now feature engineering

#loop through both datasets
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1

    # now update to no/0 if family size is greater than 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    #create feature that takes just the title of the person, eg. Mr or Miss
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    #break fare into 4 buckets, which an equal number of people in each bucket
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #split age into 5 buckets
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    print(data1['Title'].value_counts())

    stat_min = 10

    #this will create a true false series with title name as index
    title_names = (data1['Title'].value_counts() < stat_min)

    #replace small sample size title with misc
    data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(data1['Title'].value_counts())
    print("-"*10)


    data1.info()
    data_val.info()
    data1.sample(10)

    #done feature engineering, next converting the formats of categorical data for machine learning
    
