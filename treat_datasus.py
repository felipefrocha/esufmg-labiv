import pandas as pd
import numpy as np
import os
import csv
from unidecode import unidecode

datasus_path = os.getcwd()


def read_db_file_csv(path):
    dataframe = None
    with open(file=path, encoding='ISO-8859-1') as datasus_csv:
        dataframe = pd.read_csv(datasus_csv, delimiter=';', low_memory=False)
    return dataframe.copy()


def read_csv_municipios():
    cod_municipio = None
    with open(file=f'{datasus_path}/data/cod_municipio.csv', mode='r', newline='') as file_cod_municipio:
        cod_municipio = pd.read_csv(file_cod_municipio, delimiter=',', low_memory=False)
        cod_municipio['Código Município Completo'] = cod_municipio['Código Município Completo'].apply(
            lambda x: int(str(x)[0:6]))
        cod_municipio['Nome_Município'] = cod_municipio['Nome_Município'].apply(lambda x: unidecode(x).upper())
        cod_municipio['Nome_UF'] = cod_municipio['Nome_UF'].apply(lambda x: unidecode(x).upper())
        cod_municipio.rename(
            columns={'Nome_Município': 'nome_municipio', 'Código Município Completo': 'cod_municipio', 'Nome_UF': 'uf',
                     'UF': 'cod_uf'}, inplace=True)
        cod_municipio = cod_municipio[['cod_municipio', 'nome_municipio']]
    return cod_municipio.copy()


def read_csv_uf():
    cod_uf = None
    with open(file=f'{datasus_path}/data/cod_uf.csv', mode='r', newline='') as file_cod_uf:
        cod_uf = pd.read_csv(file_cod_uf, delimiter=',', low_memory=False)
        cod_uf = cod_uf[['cod_uf', 'sg_uf']]
    return cod_uf.copy()


# with open(file=f'{datasus_path}/new_consolided_file.csv', mode='w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     new_datasus = [read_db_file_csv(f'{datasus_path}/{sus_file}') for sus_file in
#                    [filename for filename in os.listdir(path=datasus_path) if filename.endswith('.csv')]]
#
#     reformat_datasus = []
#
#     datasus_length = len(new_datasus) - 1
#     datasus_columns = new_datasus[datasus_length].columns
#     reference = new_datasus[datasus_length]
#     for dataframe in new_datasus:
#         reformat_datasus.append(reference.difference(datasus_columns))


if __name__ == "__main__":
    with open(file=f'{datasus_path}/consolidated_datasus.csv', mode='w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        files = [filename for filename in os.listdir(path=f'{datasus_path}/data/datasus') if filename.endswith('.csv')]
        files.sort()
        print(files)
        new_datasus = [read_db_file_csv(f'{datasus_path}/data/datasus/{sus_file}') for sus_file in files]

        reformat_datasus = []

        datasus_length = len(new_datasus) - 1
        datasus_columns = new_datasus[datasus_length].columns

        [print(f'Tamanho do arquivo {files[i]}: {len(new_datasus[i])}') for i in range(datasus_length + 1)]

        merged_datasus = None
        for i in range(datasus_length + 1):
            if i >= datasus_length:
                merged_datasus = merged_datasus.append(new_datasus[i])
                print(f'Fim dos arquivos {i}')
                break

            difference = set(new_datasus[i].columns) ^ (set(new_datasus[i + 1].columns))

            new_datasus[i] = new_datasus[i].drop(columns=difference, errors='ignore')
            new_datasus[i + 1] = new_datasus[i + 1].drop(columns=difference, errors='ignore')

            if i == 0:
                merged_datasus = new_datasus[i].copy()
            else:
                merged_datasus = merged_datasus.append(new_datasus[i], ignore_index=True)

            print(f'{i}, {i + 1}: length {len(new_datasus[i])}, {len(new_datasus[i + 1])}')
            # print(difference)
            # print(set(new_datasus[i].columns) ^ (set(new_datasus[i+1].columns)))
            print(f'Size merged {len(merged_datasus)} Index file {i}')
            print('')

        print(f'Colunas {merged_datasus.columns}')
        print(f'Tamanho do banco do SUS: {len(merged_datasus)}')
        print(f'10 Primeiras linhas:\n {merged_datasus.head(10)}')
        print(f'10 Ultimas linhas:\n {merged_datasus.tail(10)}')

        cod_municipio = read_csv_municipios()
        cod_uf = read_csv_uf()

        print(cod_municipio.head(10))
        print(cod_uf.head(10))

        merged_datasus['SG_UF'].replace(dict(zip(cod_uf.cod_uf, cod_uf.sg_uf)), inplace=True)
        merged_datasus['ID_MUNICIP'].replace(dict(zip(cod_municipio.cod_municipio, cod_municipio.nome_municipio)),
                                             inplace=True)

        merged_datasus['DT_NOTIFIC'] = merged_datasus['DT_NOTIFIC'].apply(
            lambda x: str(x)[3:])

        print(merged_datasus[['SG_UF', 'ID_MUNICIP']].head(10))

        merg

        merged_datasus.to_csv(path_or_buf=csvfile, index=False)

    # with open(file=f'{os.getcwd()}/headers19') as headers:
    #     read = (headers)
