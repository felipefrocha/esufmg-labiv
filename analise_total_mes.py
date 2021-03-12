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
from sklearn.metrics import accuracy_score

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/staged_data/dataliked.csv') 
X2=df.iloc[:,[27]].values
Y2=df.iloc[:,[28]].values
data_index = df['DT_NOTIFIC'];
df.set_index(['DT_NOTIFIC','ID_MUNICIP'], inplace=True)

print(df.shape)
df.describe().transpose()
df = df.drop(columns=['SG_UF_NOT', 'VALOR_MIN','VALOR_MAX','EVOLUCAO'])
target_column = ['MORTALIDADE'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()


X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
print(X_train.shape); print(X_test.shape)


reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)
y_pred=reg.predict(X_test)
print(r2_score(y_pred, y_test))

###########
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
model = LinearRegression()
X_poly = poly_reg.fit_transform(X2)
fit = model.fit(X_poly, Y2)

plt.scatter(X2, Y2, color='red')
#plt.plot(X2, fit.predict(poly_reg.fit_transform(X2)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Mortalidade')
plt.ylabel('Variavel')
plt.show()
# fit the model


    
    
print(len([i for i in Y2 if i > 0.5*10**8]))
print(len([i for i in Y2 if i < 0.5*10**8]))

print(len([i for i in X2 if i > 0.5]))
print(len([i for i in X2 if i < 0.5]))

# get importance
importance = model.coef_
for i,v in enumerate(importance.T):
	print('Feature: %s, Score: %.5f' % (predictors[i],v))



model = DecisionTreeRegressor(random_state=0)
# fit the model

tree =  model.fit(X, y)
importance = tree.feature_importances_
# get importance
soma = sum(importance)
for i,v in enumerate(importance.T):
	print('Feature: %s, Score: %.5f' % (predictors[i],v))

pyplot.bar([x for x in range(len(importance.T))], importance)
pyplot.show()

####
#from sklearn.datasets import make_classification
#from sklearn.linear_model import LogisticRegression
#from matplotlib import pyplot
## define dataset
## define the model
#model = LogisticRegression()
## fit the model
#fit = model.fit(X, y)
#pred = fit.predict(X_test);
#predLog = accuracy_score(y_test, pred, normalize = True);
#print("\nLogisticRegression accuracy : ",predLog)
#
## get importance
#importance = model.coef_[0]
## summarize feature importance
#for i,v in enumerate(importance.T):
#	print('Feature: %s, Score: %.5f' % (predictors[i],v))
## plot feature importance
#pyplot.bar([x for x in range(len(importance.T))], importance)
#pyplot.show()

#media mensal por ano de cada cidade