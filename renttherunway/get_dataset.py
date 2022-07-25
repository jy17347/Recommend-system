import pandas as pd
import numpy as np

def load_dataset():
    dataset_dir = "dataset/rent/renttherunway_final_data.csv"
    loaded_dataset = pd.read_csv(dataset_dir)
    

    loaded_dataset = loaded_dataset.to_numpy()
    loaded_user = loaded_dataset[:, 2:9]
    loaded_item = loaded_dataset[:, 10:13]
    loaded_label = loaded_dataset[:, 14:16]
    
    return loaded_user, loaded_item, loaded_label

def get_trainset(loaded_user, loaded_item, loaded_label):
    user_input, item_input, labels = [],[],[]
        
    for i in range(len(loaded_user[:,0])):
        for j in range(len(loaded_item)):
            user_input.append(loaded_user[i,:])
            item_input.append(loaded_item[j])
            labels.append(loaded_label[i,j])
    
    return np.array(user_input), np.array(item_input), np.array(labels)


def scaler_user(user_dataset):

    user_dataset = user_dataset[:,1:7].astype(np.float64)
    user_dataset[:,0] = (user_dataset[:,0]-170)/30
    user_dataset[:,1] = (user_dataset[:,1]-70)/30
    user_dataset[:,2] = (user_dataset[:,2])/2
    user_dataset[:,3] = (user_dataset[:,3]-30)/30
    user_dataset[:,4] = (user_dataset[:,4]-25)/25
    user_dataset[:,5] = user_dataset[:,5]/3

    return user_dataset
