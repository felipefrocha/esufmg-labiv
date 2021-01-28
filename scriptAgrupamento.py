# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:45:01 2020

@author: tiago
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
enc = OneHotEncoder(handle_unknown='ignore')


import pandas as pd
size = 10;
ano_inicio = 2009
teste2 = [0] * size
pd.options.display.float_format = '{:,.0f}'.format
colsEnum=['CS_RACA','CS_SEXO','CS_GESTANT','CRITERIO']
dadosTratadas = []

dados = pd.read_csv("data/datasus/influd09_limpo_final.csv",sep=';',encoding = "ISO-8859-1")

for num,col in enumerate(colsEnum, start=1):
    enum = pd.read_csv("data/{}.csv".format(col),sep=';',encoding = "ISO-8859-1").columns
    dadosIter = dados
    enumTipo = dados[col]
    dadosIter[col] = dadosIter[col].fillna(9)
    dadosIter[col] = dadosIter[col].astype('category')
    
    
    
    enc_df = pd.DataFrame(enc.fit_transform(dadosIter[[col]]).toarray())
    enc_df.columns = enum
    dadosIter = dadosIter.join(enc_df)
    
    dicionario = dict([(i, ['sum'])for i in enum])
    rnm_cols = dict(sum='Sum')
    grouped_single = dadosIter.groupby(['ID_MUNICIP']).agg(dicionario).rename(columns=rnm_cols)


    grouped_single.columns = enum 
    dadosTratadas.append(grouped_single)

    
  

#horizontal_stack = pd.concat([survey_sub, survey_sub_last10], axis=1)
df_final = reduce(lambda left,right: pd.merge(left,right,on='ID_MUNICIP'), dadosTratadas).astype(int)
df_final.reset_index(level=0, inplace=True)
#result = pd.concat(teste2)
df_final.to_csv(r'data/dados_agrupados.csv', index = False,sep=';')