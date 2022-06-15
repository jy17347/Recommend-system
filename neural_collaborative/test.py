import os
import time
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import get_trainset
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model

created_time = int(time.time()) 
loss = 'binary_crossentropy'
model_name = os.listdir('C:/project/kwix/curation/neural_collaborative/model/')[-1]
model = load_model(f'model/{model_name}/exercise_model.h5')

user_dataset = load_dataset('test_user')
user_dataset = scaler_user(user_dataset)
chest_list, chest_dataset = load_dataset('test_exercise')
user_testset, chest_testset, chest_label = get_trainset(user_dataset, chest_list, chest_dataset)

binary_crossentropy = model.evaluate([user_testset, chest_testset], chest_label,batch_size = 1)
