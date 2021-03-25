# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:00:10 2021

@author: tiago
"""
import logging
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
from sklearn.preprocessing import normalize

from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import FormatStrFormatter
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#import skfuzzy as fuzz
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

def analise_anual(ano_analise:int):
	logger.info(f'INITIALIZING ANALISYS: {ano_analise}')
	df = pd.read_csv('data/staged_data/dataliked.csv') 
	colunaNome = ['TOTAL_CASOS']

	dicionario = dict([(i, ['sum'])for i in colunaNome])

	df['DT_NOTIFIC'] = df['DT_NOTIFIC'].map(lambda x: int(str(x)[-4:]))
	df = df[df['DT_NOTIFIC'] == ano_analise]
	df = df.drop(columns=['DT_NOTIFIC'])

	df['TOTAL_CASOS'] = (df['MASCULINO'] + df['IGNORADO_SEXO'] + df['FEMININO'])
	total_casos = df.groupby(['ID_MUNICIP']).agg(dicionario)


	df = df.groupby(['ID_MUNICIP']).mean()
	df['TOTAL_CASOS'] = total_casos

	df = df[(df['TOTAL_CASOS'] ) > 1 ]
	df['VALOR_TOT'] = ((df['VALOR_TOT']) / (df['TOTAL_CASOS'])).astype(float) 

	df.fillna(0)

	logger.info(df.shape)
	df.describe().transpose()
	df = df.drop(columns=['VALOR_MIN','VALOR_MAX','EVOLUCAO'])
	target_column = ['MORTALIDADE'] 

	municipios = df.index.values
	X2=df.iloc[:,[23]].values
	Y2=df.iloc[:,[24]].values

	TESTE= np.hstack((X2,Y2));

	data_teste = pd.DataFrame(TESTE);
	min_max_scaler = preprocessing.MinMaxScaler()
	data_norma = min_max_scaler.fit_transform(data_teste)
	#atributos_normal_T = pd.DataFrame(data_norm)

	##### kmeans
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(data_norma)
	y_kmeans = kmeans.predict(data_norma)

	fig, ax = plt.subplots()

	plt.scatter(X2, Y2, color='red')
	plt.scatter(TESTE[:, 0], TESTE[:, 1], c=y_kmeans, s=50, cmap='viridis')

	for i, txt in enumerate(municipios):
		ax.annotate(txt, (X2[i], Y2[i]),fontsize=1)

	plt.title(f'Cidades ano - {ano_analise} - Mortalidade X Orçamento')
	plt.xlabel('Mortalidade')
	plt.ylabel('Orçamento')



	fig.savefig(f'teste_money{ano_analise}.png',dpi=2500)


def analise_main():
	logger.info(f'INITIALIZING ANALISYS')
	df = pd.read_csv('data/staged_data/dataliked.csv') 
	colunaNome = ['TOTAL_CASOS']

	dicionario = dict([(i, ['sum'])for i in colunaNome])

	df = df.drop(columns=['DT_NOTIFIC'])

	df['TOTAL_CASOS'] = (df['MASCULINO'] + df['IGNORADO_SEXO'] + df['FEMININO'])
	total_casos = df.groupby(['ID_MUNICIP']).agg(dicionario)


	df = df.groupby(['ID_MUNICIP']).mean()
	df['TOTAL_CASOS'] = total_casos

	df = df[(df['TOTAL_CASOS'] ) > 30 ]
	df['VALOR_TOT'] = ((df['VALOR_TOT']) / (df['TOTAL_CASOS'])).astype(float) 

	df.fillna(0)

	logger.info(df.shape)
	df.describe().transpose()
	df = df.drop(columns=['VALOR_MIN','VALOR_MAX','EVOLUCAO'])
	target_column = ['MORTALIDADE'] 

	municipios = df.index.values
	X2=df.iloc[:,[23]].values
	Y2=df.iloc[:,[24]].values

	TESTE= np.hstack((X2,Y2));

	data_teste = pd.DataFrame(TESTE);
	min_max_scaler = preprocessing.MinMaxScaler()
	data_norma = min_max_scaler.fit_transform(data_teste)
	#atributos_normal_T = pd.DataFrame(data_norm)

	##### kmeans
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(data_norma)
	y_kmeans = kmeans.predict(data_norma)

	fig, ax = plt.subplots()

	plt.scatter(X2, Y2, color='red')
	plt.scatter(TESTE[:, 0], TESTE[:, 1], c=y_kmeans, s=50, cmap='viridis')

	for i, txt in enumerate(municipios):
		ax.annotate(txt, (X2[i], Y2[i]),fontsize=1)

	plt.title(f'Cidades, media total - Mortalidade X Orçamento')
	plt.xlabel('Mortalidade')
	plt.ylabel('Orçamento')



	fig.savefig(f'teste_money.png',dpi=2500)

