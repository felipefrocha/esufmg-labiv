"""
@author: Felipe Fonseca Rocha
"""

import csv
import logging
import multiprocessing
import os
from datetime import date
from typing import Tuple, List

import pandas as pd
from unidecode import unidecode


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


current_director = os.getcwd()
municipios_file_path = 'data/cod_municipio.csv'
uf_file_path = 'data/cod_uf.csv'

colmuns_city = {
    'Nome_Município': 'nome_municipio',
    'Código Município Completo': 'cod_municipio',
    'Nome_UF': 'uf',
    'UF': 'cod_uf'
}

OBITO = 1
NAO_OBITO = 0
CURA = 1
IGNORADO = 9


def read_db_file_csv(path: str) -> pd.DataFrame:
    """
    Read data from datasus db
    :param path:
    :return:
    """
    dataframe = None
    with open(file=path, encoding='ISO-8859-1') as datasus_csv:
        dataframe = pd.read_csv(datasus_csv, delimiter=';', low_memory=False)
    return dataframe.copy()


def data_reader(path: any, data: List) -> List[pd.DataFrame]:
    """
    Retrieve data from file .csv
    :param path:
    :param data:
    :return:
    """
    if type(path) is str:
        with open(file=f'{current_director}/{path}', mode='r', newline='',
                  encoding='utf-8') as file:
            data.append(pd.read_csv(file, delimiter=',', low_memory=False))
        return data

    for file_path in path:
        with open(file=f'{current_director}/{file_path}', mode='r', newline='',
                  encoding='utf-8') as file:
            data.append(pd.read_csv(file, delimiter=',', low_memory=False))
    return data


def read_csv_municipios(path: str) -> pd.DataFrame:
    """
    Read and clean data from all cities at IBGE data
    :param path:
    :return Data frame content municipio data:
    """
    data = []

    municipios = data_reader(path, data)[0]

    municipios = clean_cities_data(municipios)

    return municipios.copy()


def read_csv_uf(path: str) -> pd.DataFrame:
    """
    Read and clean data from all cities at IBGE data
    :param path:
    :return:
    """
    data = []
    cod_uf = data_reader(path, data)[0]
    cod_uf = clean_states_data(cod_uf)
    return cod_uf.copy()


def clean_cities_data(cities: pd.DataFrame) -> pd.DataFrame:
    """
    Cities retriving only cod and name
    :param cities: 
    :return: 
    """
    cities['Código Município Completo'] = cities['Código Município Completo'].apply(
        lambda x: int(str(x)[0:6]))
    cities['Nome_Município'] = cities['Nome_Município'].apply(lambda x: unidecode(x).upper())
    cities['Nome_UF'] = cities['Nome_UF'].apply(lambda x: unidecode(x).upper())
    cities.rename(
        columns=colmuns_city, inplace=True)
    cities = cities[['cod_municipio', 'nome_municipio']]
    return cities


def clean_states_data(cod_uf: pd.DataFrame) -> pd.DataFrame:
    """
    Remove undesired columns
    :param cod_uf:
    :return:
    """
    new_uf = cod_uf[['cod_uf', 'sg_uf']]
    return new_uf.copy()


def string2date(data: str) -> date:
    """
    Convert string date format dd/mm/yyyy to date
    :param data:
    :return:
    """
    x = data.split(sep='/')
    day = int(x[0])
    month = int(x[1])
    year = int(x[2])
    return date(year, month, day)


def days_between(date_initial: str, date_final: str) -> int:
    """
    Subtraction 2 dates strings (dd/mm/yyyy)
    :param date_initial:
    :param date_final:
    :return:
    """
    try:
        d0 = string2date(date_initial)
        d1 = string2date(date_final)
    except:
        return 0
    delta = d1 - d0
    return delta.days


def get_filter_datasus_columns():
    return [
        "DT_NOTIFIC",
        "ID_MUNICIP",
        "SG_UF_NOT",
        "CS_SEXO",
        "CS_RACA",
        "CS_GESTANT",
        "CS_ESCOL_N",
        "PNEUMOPATI",
        "HOSPITAL",
        "DT_INTERNA",
        "DT_EVOLUCA",
        "CRITERIO",
        "DOENCA_TRA",
        "EVOLUCAO",
        "UTI",
    ]


def get_location_code_data(cities_path: str, uf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return read_csv_municipios(cities_path), read_csv_uf(uf_path)


def consolidate_datasus():
    with open(file=f'{current_director}/consolidated_datasus.csv', mode='w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        datasus_length, new_datasus = funout_datasus_treat()

        merged_datasus = None
        for i in range(datasus_length + 1):
            if i >= datasus_length:
                merged_datasus = merged_datasus.append(new_datasus[i])
                logger.info(f'Fim dos arquivos {i}')
                break

            difference = set(new_datasus[i].columns) ^ (set(new_datasus[i + 1].columns))

            new_datasus[i] = new_datasus[i].drop(columns=difference, errors='ignore')
            new_datasus[i + 1] = new_datasus[i + 1].drop(columns=difference, errors='ignore')

            if i == 0:
                merged_datasus = new_datasus[i].copy()
            else:
                merged_datasus = merged_datasus.append(new_datasus[i], ignore_index=True)

            logger.info(f'{i}, {i + 1}: length {len(new_datasus[i])}, {len(new_datasus[i + 1])}')
            # logger.info(difference)
            # logger.info(set(new_datasus[i].columns) ^ (set(new_datasus[i+1].columns)))
            logger.info(f'Size merged {len(merged_datasus)} Index file {i}')
            logger.info('')

        # Log Brief data
        logger.info(f'Colunas {merged_datasus.columns}')
        logger.info(f'Tamanho do banco do SUS: {len(merged_datasus)}')
        logger.info(f'10 Primeiras linhas:\n {merged_datasus.head(10)}')
        logger.info(f'10 Ultimas linhas:\n {merged_datasus.tail(10)}')

        # Retrieve data From auxiliar tables UF and Municipio
        cod_municipio, cod_uf = get_location_code_data(municipios_file_path, uf_file_path)

        # Brief Data
        logger.info(cod_municipio.head(10))
        logger.info(cod_uf.head(10))

        # TODO - Get list from parameter
        # Filter columns of interest
        merged_datasus = merged_datasus[get_filter_datasus_columns()]
        # Filter state MG
        merged_datasus = merged_datasus[merged_datasus["SG_UF_NOT"] == 31]
        # Replace code UF by Code UF Name
        merged_datasus['SG_UF_NOT'].replace(dict(zip(cod_uf.cod_uf, cod_uf.sg_uf)), inplace=True)
        # Replace Code Municipio by Code Municipio Name
        merged_datasus['ID_MUNICIP'].replace(dict(zip(cod_municipio.cod_municipio, cod_municipio.nome_municipio)),
                                             inplace=True)
        # Ajusta date for consider only month
        merged_datasus['DT_NOTIFIC'] = merged_datasus['DT_NOTIFIC'].apply(
            lambda x: str(x)[3:])

        # merged_datasus['TMP_ATE_OBITO'] = merged_datasus.apply( lambda row: (
        #     sub_date( row['DT_INTERNA'], row['DT_EVOLUCA'] ) if row['EVOLUCAO'] == 1 else None),
        #                                                         axis=1 )

        logger.info(merged_datasus[['SG_UF_NOT', 'ID_MUNICIP']].head(10))

        merged_datasus = merged_datasus[merged_datasus['EVOLUCAO'] > 0]
        # merged_datasus = merged_datasus[merged_datasus['TMP_ATE_OBITO'] > 0]
        merged_datasus.head()
        logger.info(len(merged_datasus))

        merged_datasus.to_csv(path_or_buf=csvfile, index=False)

        logger.info("FINISH")
    # with open(file=f'{os.getcwd()}/headers19') as headers:
    #     read = (headers)


def run_funout_datasus_process(sus_file: str) -> Tuple[str,pd.DataFrame]:
    database_per_year = read_db_file_csv(f'{current_director}/data/datasus/{sus_file}')
    database = database_per_year.rename(columns={'DT_OBITO': 'DT_EVOLUCA'})
    database['EVOLUCAO'] = database['EVOLUCAO'].apply(
        lambda x: NAO_OBITO if x == IGNORADO or x == CURA or x is None else OBITO)
    logger.info(f'Tamanho do arquivo {sus_file}: {len(database)}')
    return sus_file, database


def funout_datasus_treat():
    files = [filename for filename in os.listdir(path=f'{current_director}/data/datasus') if filename.endswith('.csv')]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        threads = [pool.apply_async(run_funout_datasus_process, (file,)) for file in files]
        results = [res.get() for res in threads]

    results.sort(key=lambda x: x[0])

    new_datasus = [v for k, v in results]

    datasus_length = len(new_datasus) - 1

    return datasus_length, new_datasus
