# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:00:10 2021

@author: tiago
"""

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv('data/staged_data/dataliked.csv') 
df.set_index(['DT_NOTIFIC'], inplace=True)
print(df.shape)
df.describe().transpose()
df = df.drop(columns=['SG_UF_NOT', 'ID_MUNICIP','VALOR_MIN','VALOR_MAX','EVOLUCAO'])
target_column = ['MORTALIDADE'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()


X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)


reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)
y_pred=reg.predict(X_test)
print(r2_score(y_pred, y_test))

###########

model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
#for i,v in enumerate(importance.T):
	#print('Feature: %s, Score: %.5f' % (predictors[i],v))
# plot feature importance






###
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# define dataset
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance.T):
	print('Feature: %s, Score: %.5f' % (predictors[i],v))
# plot feature importance
pyplot.bar([x for x in range(len(importance.T))], importance)
pyplot.show()