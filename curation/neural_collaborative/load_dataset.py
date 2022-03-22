import os
import pandas as pd
import numpy as np

def load_numpy_dataset(data):
    dataset_dir = "C:/project/dataset/kwix/curation"
    loaded_dataset = pd.read_csv(dataset_dir +'/' + data + '.csv')
    
    if data == 'user':
        loaded_dataset.iloc[:,1] = (loaded_dataset.iloc[:,1]-170)/30
        loaded_dataset.iloc[:,2] = (loaded_dataset.iloc[:,2]-70)/30
        loaded_dataset.iloc[:,4] = (loaded_dataset.iloc[:,4]-30)/30
        loaded_dataset.iloc[:,5] = (loaded_dataset.iloc[:,5]-30)/20
        loaded_dataset = loaded_dataset.to_numpy()[:,1:]
        return loaded_dataset
    
    else:
        exercise_list = np.array(loaded_dataset.columns[1:],dtype = int)
        loaded_dataset = loaded_dataset.to_numpy()[:,1:]
        return exercise_list, loaded_dataset

def get_trainset(user_profile, exercise, labeling):
    user_input, item_input, labels = [],[],[]
        
    for i in range(len(user_profile[:,0])):
        for j in range(len(exercise)):
            user_input.append(user_profile[i,:])
            item_input.append(exercise[j])
            labels.append(labeling[i,j])
    
    return np.array(user_input), np.array(item_input), np.array(labels)
