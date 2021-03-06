import os
import numpy as np
import pandas as pd
import tensorflow as tf
from user_info import person
from user_info import input_bodyInfo
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model

exer_name = pd.read_csv('C:/project//kwix/curation/dataset/exercise_list.csv')

model_name = os.listdir('C:/project/kwix/curation/neural_collaborative/model/')[-1]
model = load_model(f'model/{model_name}/exercise_model.h5')

chest_list, temp = load_dataset('exercise')

name = 'kjw'
weight = 72
height = 180
sex = 1
age = 26
ability = 3
# name, height, weight, sex, age = input_bodyInfo
newbie = person(name, height, weight, sex, age, ability)
newbie.check()
print(newbie.profile())
newbie_input = scaler_user(newbie.profile())
print((newbie_input))

recommend_probability = []
for i in range(len(chest_list)):
    recommend_predict = model.predict([newbie_input, np.array([chest_list[i]])])
    print(np.around(recommend_predict,2),end=' ')
    recommend_probability.append(recommend_predict[0,0])
a=np.array(recommend_probability).argsort()
# print(a)
print("")
topK = 4
reco_topK = []
for i in range(topK):
    reco_topK.append(exer_name.iloc[3,a[::-1][i]+1])

print(reco_topK)