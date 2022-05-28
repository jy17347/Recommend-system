import numpy as np
import pandas as pd
import tensorflow as tf
from user_info import person
from user_info import input_bodyInfo
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model
model_name = input("model:")
loss = model_name
exer_name = pd.read_csv('C:/project/dataset/kwix/curation/exercise_list.csv')
# model = load_model(f'model/chest_model_{loss}.h5')
model = load_model(f'model/binary_crossentropy_1649318650/chest_model.h5') 
chest_list, temp = load_dataset('abdominal')

print(model.layers[5].get_weights)

print(exer_name)
name = 'ww'
weight = 78
height = 181
sex = 1
age = 26
ability = 4
# name, height, weight, sex, age = input_bodyInfo
newbie = person(name, height, weight, sex, age, ability)
newbie.check()
print(newbie.profile())
newbie_input = scaler_user(newbie.profile())
print((newbie_input))

recommend_probability = []
for i in range(len(chest_list)):
    recommend_predict = model.predict([newbie_input, np.array([chest_list[i]])])
    print(recommend_predict)
    recommend_probability.append(recommend_predict[0,0])
a=np.array(recommend_probability).argsort()
print(a)
print()
topK = 4
reco_topK = []
for i in range(topK):
    reco_topK.append(exer_name.iloc[0,a[::-1][i]+1])

print(reco_topK)