# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:45:01 2020

@author: tiago
"""
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')


import pandas as pd
pd.options.display.float_format = "{:.2f}".format
size = 10;
ano_inicio = 2009
teste2 = [0] * size

racasEnum = ['nulo','branca','preta','amarela','parda','indígena','ignorado']
   
orcamento_mg = pd.read_csv("data/datasus/influd09_limpo_final.csv",sep=';',encoding = "ISO-8859-1")
    
racas = orcamento_mg['CS_RACA']
orcamento_mg['CS_RACA'] = orcamento_mg['CS_RACA'].fillna(0)
orcamento_mg['CS_RACA'] = orcamento_mg['CS_RACA'].astype('category')



enc_df = pd.DataFrame(enc.fit_transform(orcamento_mg[['CS_RACA']]).toarray())
enc_df.columns = teste
orcamento_mg = orcamento_mg.join(enc_df)
grouped_single = orcamento_mg.groupby(['ID_MUNICIP']).agg({'branca': ['sum'],'preta':['sum']})
rnm_cols = dict(sum='Sum')
grouped_single = orcamento_mg.set_index(teste).stack().groupby('ID_MUNICIP').agg(rnm_cols.keys()).rename(columns=rnm_cols)

    
       


result = pd.concat(teste2)
result.to_csv(r'data/orcamento/orcamento_mg_consolidado.csv', index = False,sep=';')