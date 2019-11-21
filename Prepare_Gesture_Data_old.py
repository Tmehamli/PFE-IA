import os
import numpy as np
import pandas as pd
from python_linq import From

default_path = 'data/'
default_file = 'a1_raw.csv'
seed = 42
np.random.seed(seed=seed)

def input_selection(df_file, inputs_to_keep=[]) :
    if len(inputs_to_keep)!=0:
        #print(inputs_to_keep)
        for header in df_file.columns.to_list():
            if header not in inputs_to_keep: 
                del df_file[header]
                
def creat_missing_values(df_file, nb_values=1):
    nb_row = df_file.shape[0]
    if nb_values < nb_row:
        missing_index = np.random.randint(nb_values, size=nb_values)

def prepare_data(path=default_path, file=default_file):
    df_file = pd.read_csv(path + file, sep=',')
    # TODO : inputs_to_keep define by caller
    inputs_to_keep = df_file.columns.to_list()[0:1] + df_file.columns.to_list()[-2:-1]
    input_selection(df_file, inputs_to_keep)
    #To test if load_data works
    dataFrames = load_data(default_path)
    try:
        df_file.to_pickle(path + file[:-4]+".pkl")
    except:
        raise Error('Problem with DataFrame.to_pickle()')
        
def load_data(path):
    all_files = os.listdir(path)
    files_to_load = From(all_files).where(lambda file: file[-3:]=='csv').where(lambda file: file[-5]=='w').toList()
    df_Big = pd.read_csv(path + files_to_load[0], sep=',')
    for file in files_to_load[1:]:
        df_file = pd.read_csv(path + file, sep=',')
        df_Big = df_Big.append(df_file)
    grouped = df_Big.groupby(['phase'])
    dataFrames = {}
    for phase, group in grouped:
        if phase != 'Preparação':
            tmp_group = group.iloc[lambda row: row.index%30==0]
            ## Don't know if data have to be sorted or not
            #dataFrames['df_'+phase] = tmp_group.sort_values(by=['timestamp']).reset_index(drop=True)
            dataFrames['df_'+phase] = tmp_group.reset_index(drop=True)
    return dataFrames

#expected = {'Rest':65, 'Preparation':115, 'Hold':76, 'Stroke':49, 'Retractation':73}
#From(dataFrames).select(lambda df: dataFrames[df].shape[0]).toList()