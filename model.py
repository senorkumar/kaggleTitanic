import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
