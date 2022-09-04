import pandas as pd
import numpy as np

def load_dataset(data):
    dataset_dir = "./dataset"
    loaded_dataset = pd.read_csv(dataset_dir +'/' + data + '.csv')
    
    if data in ['user', 'test_user']:
        loaded_user = loaded_dataset.to_numpy()
        return loaded_user
    
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


def scaler_user(user_dataset):

    user_dataset = user_dataset[:,1:12].astype(np.float64)
    user_dataset[:,0] = (user_dataset[:,0]-36.7)/28
    user_dataset[:,2] = (user_dataset[:,2]-168.5)/44
    user_dataset[:,3] = (user_dataset[:,3]-70)/70
    user_dataset[:,4] = (user_dataset[:,4]-23.2)/56
    user_dataset[:,5] = (user_dataset[:,5]-79)/79
    user_dataset[:,6] = (user_dataset[:,6]-131)/120
    user_dataset[:,7] = (user_dataset[:,7]-37)/37
    user_dataset[:,8] = (user_dataset[:,8]-15)/70
    user_dataset[:,9] = (user_dataset[:,9]-40)/40
    user_dataset[:,10] = (user_dataset[:,10]-191)/191  
    
    return user_dataset

user = scaler_user(load_dataset('user'))
print(np.shape(user))