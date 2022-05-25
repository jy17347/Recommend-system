import numpy as np
import pandas as pd
import tensorflow as tf
import os
from user_info import person
from user_info import input_bodyInfo
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model
exer_name = pd.read_csv('C:/project/dataset/kwix/curation/exercise_list.csv')
model_name = os.listdir('C:/project/kwix/curation/neural_precisionK/model/')[-1]
model = load_model(f'model/{model_name}/chest_model.h5')
chest_list, temp = load_dataset('chest')
print(model.layers[5].get_weights)

print(exer_name)

# name = 'ww'
# weight = 78
# height = 180
# sex = 1
# age = 26
# ability = 4

name, height, weight, sex, age, ability = input_bodyInfo()

newbie = person(name, height, weight, sex, age, ability)
newbie.check()
print(newbie.profile())
newbie_input = scaler_user(newbie.profile())
print((newbie_input))

recommend_predict = model.predict([newbie_input, np.array([chest_list])])
print(f'{recommend_predict}')
print(np.around(recommend_predict.flatten(),3))
