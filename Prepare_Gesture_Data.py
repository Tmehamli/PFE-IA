import os
import numpy as np
import pandas as pd

def input_selection(df_file, inputs_to_keep=[]) :
    if len(inputs_to_keep)!=0:
        #print(inputs_to_keep)
        for header in df_file.columns.to_list():
            if header not in inputs_to_keep: 
                del df_file[header]
default_path = 'data/'
default_file = 'a1_raw.csv'

def prepare_data(path=default_path, file=default_file):
    df_file = pd.read_csv(path + file, sep=',')
    # TODO : inputs_to_keep define by caller
    inputs_to_keep = df_file.columns.to_list()[0:1] + df_file.columns.to_list()[-2:-1]
    input_selection(df_file, inputs_to_keep)
    try:
        df_file.to_pickle(path + file[:-4]+".pkl")
    except:
        raise Error('Problem with DataFrame.to_pickle()')
