import os
import pandas as pd
import numpy as np

def load_num_dataset(data):
    dataset_dir = "C:/project/dataset/kwix/curation"
    loaded_dataset = pd.read_csv(dataset_dir +'/' + data + '.csv')
    
    if data == 'user':
        loaded_dataset.iloc[:,1] = (loaded_dataset.iloc[:,1]-170)/30
        loaded_dataset.iloc[:,2] = (loaded_dataset.iloc[:,2]-70)/30
        loaded_dataset.iloc[:,4] = (loaded_dataset.iloc[:,4]-30)/30
        loaded_dataset.iloc[:,5] = (loaded_dataset.iloc[:,5]-30)/20
    loaded_dataset = loaded_dataset.to_numpy()[:,1:]
    return loaded_dataset