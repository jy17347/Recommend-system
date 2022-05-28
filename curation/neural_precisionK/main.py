import numpy as np
import pandas as pd
import tensorflow as tf
import os
from user_info import person
from user_info import input_bodyInfo
from get_dataset import load_list
from get_dataset import scaler_user
from tensorflow.keras.models import load_model

exercise_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
model_name = os.listdir('C:/project/kwix/curation/neural_precisionK/model/')[-1]
model = load_model(f'model/{model_name}/exercise_model.h5')

abdominal_list = load_list('abdominal')
chest_list = load_list('chest')
lower_list = load_list('lower')

print(model.layers[5].get_weights)

name = 'ww'
weight = 78
height = 180
sex = 1
age = 26
during = 3

# name, height, weight, sex, age, during = input_bodyInfo()

newbie = person(name, height, weight, sex, age, during)
newbie.check()
print(newbie.profile())
newbie_input = scaler_user(newbie.profile())
print((newbie_input))

recommend_predict = model.predict([newbie_input, np.array([exercise_list])])
print(f'{recommend_predict}')
recommend_predict = np.around(recommend_predict.flatten(),3)
top_K = 4

a=recommend_predict.argsort()[::-1]

for i in a[:4]:
    print(f'1. {chest_list[i]},\t 2. {abdominal_list[i]},\t 3. {lower_list[i]}')
