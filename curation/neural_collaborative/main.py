import numpy as np
import tensorflow as tf
from get_dataset import load_dataset
from user_info import input_bodyInfo
from user_info import person
from get_dataset import scaler_user
from tensorflow.keras.models import load_model

model = load_model('model/curation_model.h5')
chest_list, temp = load_dataset('chest')

name = 'ww'
weight = 75
height = 181
sex = 1
age = 25
# name, height, weight, sex, age = input_bodyInfo
newbie = person(name, height, weight, sex, age)
newbie.check()
print(newbie.profile())
newbie_input = scaler_user(newbie.profile())
print((newbie_input))

recommend_list = []
for i in range(len(chest_list)):
    recommend_predict = model.predict([newbie_input, np.array([chest_list[i]])])
    print(recommend_predict)
    recommend_list.append(recommend_predict[0,0])

a=np.array(recommend_list).argsort()
print(a)