# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:45:01 2020

@author: tiago
"""
import logging

import pandas as pd

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

def consolidate_budget():

    pd.options.display.float_format = "{:.2f}".format
    size = 10;
    ano_inicio = 2009
    teste2 = [0] * size
    for x in range(size):
        orcamento_mg = pd.read_csv("data/orcamento/orcamento_mg_{}.csv".format(ano_inicio), sep=';')

        orcamento_mg["Valor Total"] = orcamento_mg["Valor Total"].replace('[.]', '', regex=True).replace('[,]', '.',
                                                                                                         regex=True).replace(
            '[R$]', '', regex=True).astype(float)
        orcamento_mg["DATA OB"] = orcamento_mg["DATA OB"].apply(lambda x: x[3:10])
        group = orcamento_mg['Valor Total'].groupby(orcamento_mg['MUNICIPIO'])

        grouped_single = orcamento_mg.groupby(['MUNICIPIO', 'DATA OB']).agg({'Valor Total': ['sum', 'min', 'max']})
        grouped_single = grouped_single.reset_index()
        grouped_single.columns = ['MUNICIPIO', 'DATA OB', 'Valor Total', 'Valor Minimo', 'Valor maximo']
        grouped_single['Valor Total'] = grouped_single['Valor Total'].map('{:.2f}'.format)
        teste = group.mean();
        logger.info(group.mean())
        teste2[x] = grouped_single
        ano_inicio = ano_inicio + 1

    result = pd.concat(teste2)
    result.to_csv(r'data/staged_data/orcamento_mg_consolidado.csv', index=False, sep=';')
