# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:45:01 2020

@author: tiago
"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
enc = OneHotEncoder(handle_unknown='ignore')

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

def create_splited_columns():
    size = 10;
    ano_inicio = 2009
    pd.options.display.float_format = '{:,.0f}'.format
    colsEnum=['CS_RACA','CS_SEXO','CS_GESTANT','CRITERIO']
    colsGroup=['EVOLUCAO','PNEUMOPATI','UTI','HOSPITAL']
    dadosTratadas = []

    dados = pd.read_csv("data/staged_data/consolidated_datasus.csv",sep=',',encoding = "ISO-8859-1")
    for num,col in enumerate(colsEnum, start=1):
        logger.info(col)
        enum = pd.read_csv("data/{}.csv".format(col),sep=';',encoding = "ISO-8859-1").columns
        dadosIter = dados
        enumTipo = dados[col]
        dadosIter[col] = dadosIter[col].fillna(9)
        dadosIter[col] = dadosIter[col].astype('category')



        enc_df = pd.DataFrame(enc.fit_transform(dadosIter[[col]]).toarray())
        enc_df.columns = enum
        dadosIter = dadosIter.join(enc_df)

        dicionario = dict([(i, ['sum'])for i in enum])
        print(dicionario)
        rnm_cols = dict(sum='Sum')
        grouped_single = dadosIter.groupby(['ID_MUNICIP','DT_NOTIFIC','SG_UF_NOT']).agg(dicionario).rename(columns=rnm_cols)
        

        grouped_single.columns = enum
        dadosTratadas.append(grouped_single)
        
    for num,col in enumerate(colsGroup, start=1):
        colunaNome=[col]
        logger.info(col)
        dadosIter = dados
        dadosIter[col] = dadosIter[col].fillna(9)
        print(col)
        print(colunaNome)
        
        dicionario2 = dict([(i, ['sum'])for i in colunaNome])
        print(dicionario)
        rnm_cols = dict(sum='Sum')
        grouped_single = dadosIter.groupby(['ID_MUNICIP','DT_NOTIFIC','SG_UF_NOT']).agg(dicionario2).rename(columns=rnm_cols)

        grouped_single.columns = colunaNome
        dadosTratadas.append(grouped_single)
    
    #horizontal_stack = pd.concat([survey_sub, survey_sub_last10], axis=1)
    df_final = reduce(lambda left,right: pd.merge(left,right,on=['ID_MUNICIP','DT_NOTIFIC','SG_UF_NOT']), dadosTratadas).astype(int)
    df_final.reset_index(level=0, inplace=True)
    df_final['MORTALIDADE'] = ((df_final['EVOLUCAO']) / (df_final['MASCULINO'] + df_final['IGNORADO_SEXO'] + df_final['FEMININO'])).astype(float)  
    df_final.to_csv(r'data/staged_data/dados_agrupados.csv', index = True,sep=',')

