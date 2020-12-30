import pandas as pd
import numpy as np
import os
import csv

datasus_path = os.getcwd()


def read_db_file_csv(path):
    dataframe = None
    with open(file=path, encoding='ISO-8859-1') as datasus_csv:
        dataframe = pd.read_csv(datasus_csv, delimiter=';', low_memory=False)
    return dataframe.copy()


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

        [print(f'Tamanho do arquivo {files[i]}: {len(new_datasus[i])}') for i in range(datasus_length+1)]

        merged_datasus= None
        for i in range(datasus_length+1):
            if i >= datasus_length:
                merged_datasus = merged_datasus.append(new_datasus[i])
                print(f'Fim dos arquivos {i}')
                break

            difference = set(new_datasus[i].columns) ^ (set(new_datasus[i+1].columns))

            new_datasus[i] = new_datasus[i].drop(columns=difference, errors='ignore')
            new_datasus[i+1] = new_datasus[i+1].drop(columns=difference, errors='ignore')

            if i == 0:
                merged_datasus = new_datasus[i].copy()
            else:
                merged_datasus = merged_datasus.append(new_datasus[i],ignore_index=True)


            print(f'{i}, {i+1}: length {len(new_datasus[i])}, {len(new_datasus[i+1])}')
            # print(difference)
            # print(set(new_datasus[i].columns) ^ (set(new_datasus[i+1].columns)))
            print(f'Size merged {len(merged_datasus)} Index file {i}')
            print('')

        print(f'Colunas {merged_datasus.columns}')
        print(f'Tamanho do banco do SUS: {len(merged_datasus)}')
        print(f'10 Primeiras linhas:\n {merged_datasus.head(10)}')
        print(f'10 Ultimas linhas:\n {merged_datasus.tail(10)}')

        merged_datasus.to_csv(path_or_buf=csvfile,index=False)


    # with open(file=f'{os.getcwd()}/headers19') as headers:
    #     read = (headers)


