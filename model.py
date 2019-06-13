import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import seaborn as sns


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#working through https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy to learn more
