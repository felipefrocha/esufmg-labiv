# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:00:10 2021

@author: tiago
"""

# Import required libraries
import logging
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

###
# Configure logs
###
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
###
# END - Configure logs
###

def analise_reg_por_ano(ano_analise:int = 2012):
	df2 = pd.read_csv('data/staged_data/dataliked.csv') 

	df2['VALOR_TOT'] = ((df2['VALOR_TOT']) / (df2['MASCULINO'] + df2['IGNORADO_SEXO'] + df2['FEMININO'])).astype(float) 

	df2['DT_NOTIFIC'] = df2['DT_NOTIFIC'].map(lambda x: int(str(x)[-4:]))

	df = df2[df2['DT_NOTIFIC'] == ano_analise].reset_index(drop=True)

	rnm_cols = dict(sum='Sum')
	df = df.groupby(['DT_NOTIFIC','ID_MUNICIP']).mean()

	df.fillna(0)

	logger.info(df.shape)
	df.describe().transpose()
	df = df.drop(columns=['VALOR_MIN','VALOR_MAX','EVOLUCAO'])
	target_column = ['MORTALIDADE'] 

	municipios = df.index.values
	X2=df.iloc[:,[23]].values
	Y2=df.iloc[:,[24]].values


	predictors = list(set(list(df.columns))-set(target_column))
	df[predictors] = df[predictors]/df[predictors].max()
	df.describe().transpose()


	X = df[predictors].values
	y = df[target_column].values

	X = np.where(np.isnan(X), 0, X)
	y = np.where(np.isnan(X), 0, y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
	logger.info(X_train.shape); logger.info(X_test.shape)


	reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)
	y_pred=reg.predict(X_test)
	logger.info(r2_score(y_pred, y_test))

	###########
	from sklearn.preprocessing import PolynomialFeatures
	from matplotlib.ticker import FormatStrFormatter

	poly_reg = PolynomialFeatures(degree=4)
	model = LinearRegression()
	X_poly = poly_reg.fit_transform(X2)
	fit = model.fit(X_poly, Y2)


	fig, ax = plt.subplots()

	plt.scatter(X2, Y2, color='red')

	for i, txt in enumerate(municipios):
		ax.annotate(txt[1], (X2[i], Y2[i]),fontsize=1)
	#plt.plot(X2, fit.predict(poly_reg.fit_transform(X2)), color='blue')
	plt.title('Truth or Bluff (Linear Regression)')
	plt.xlabel('Mortalidade')
	plt.ylabel('Variavel')


	fig.savefig(f'regressao_{ano_analise}.png',dpi=2500)
	# fit the model

	logger.info(len([i for i in Y2 if i > 0.5*10**8]))
	logger.info(len([i for i in Y2 if i < 0.5*10**8]))

	logger.info(len([i for i in X2 if i > 0.5]))
	logger.info(len([i for i in X2 if i < 0.5]))

	# get importance
	importance = model.coef_
	for i,v in enumerate(importance.T):
		logger.info('Feature: %s, Score: %.5f' % (predictors[i],v))


	model = DecisionTreeRegressor(random_state=0)
	# fit the model

	tree =  model.fit(X, y)
	importance = tree.feature_importances_
	# get importance
	soma = sum(importance)
	for i,v in enumerate(importance.T):
		logger.info('Feature: %s, Score: %.5f' % (predictors[i],v))

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
#logger.info("\nLogisticRegression accuracy : ",predLog)
#
## get importance
#importance = model.coef_[0]
## summarize feature importance
#for i,v in enumerate(importance.T):
#	logger.info('Feature: %s, Score: %.5f' % (predictors[i],v))
## plot feature importance
#pyplot.bar([x for x in range(len(importance.T))], importance)
#pyplot.show()

#media mensal por ano de cada cidade