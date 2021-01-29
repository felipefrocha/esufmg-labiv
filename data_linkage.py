import csv
import logging
import os
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


current_directory = os.getcwd()


def data_reader(path: any):
    """
    Retrieve data from file .csv
    :param path:
    :param data:
    :return:
    """
    data = None
    if type(path) is str:
        with open(file=f'{current_directory}/{path}', mode='r', newline='',
                  encoding='utf-8') as file:
            data = pd.read_csv(file, delimiter=',', low_memory=False)
    return data.copy()


def import_files():
    sus = data_reader("data/staged_data/consolidated_datasus.csv")
    budget = data_reader("data/staged_data/orcamento_mg_consolidado.csv")
    sus['DT_NOTIFIC'] = sus['DT_NOTIFIC'].map(lambda x: x.replace('/',''))
    sus['ID_MUNICIP'] = sus['ID_MUNICIP'].map(lambda x: x.replace(' ',''))
    budget['DT_NOTIFIC'] = sus['DT_NOTIFIC'].map(lambda x: x.replace('/',''))
    budget['ID_MUNICIP'] = sus['ID_MUNICIP'].map(lambda x: x.replace(' ',''))

    sus['ID'] = sus['DT_NOTIFIC'] + sus['ID_MUNICIP']
    budget['ID'] = budget['DT_NOTIFIC'] + budget['ID_MUNICIP']




    sus.set_index(['ID'], inplace=True)
    budget.set_index(['ID'], inplace=True)
    sus.sort_index(inplace=True)
    budget.sort_index(inplace=True)
    with open(file=f'{current_directory}/data/staged_data/dataliked.csv', mode='w', newline='') as csvfile:

        merged_data = pd.concat([sus, budget], axis=1, join="inner")
        logger.info(f'Teste de index: \n{sus.index.intersection(budget.index)}')
        merged_data.to_csv(path_or_buf=csvfile, index=False)

        logger.info("FINISH")

