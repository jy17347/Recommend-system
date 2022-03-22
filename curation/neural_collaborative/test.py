import numpy as np
import tensorflow as tf
from load_dataset import load_numpy_dataset
from tensorflow.keras.models import load_model
from user_info import input_bodyInfo
from user_info import person

chest_list, temp = load_numpy_dataset('chest')

model = load_model('model/curation_model.h5')

name = 'ww'
weight = 75
height = 181
sex = 1
age = 25
newbie = person(name, height, weight, sex, age)
newbie.check()

print((np.shape(newbie.user_input())))
recommend_list = []
for i in range(len(chest_list)):
    recommend_predict = model.predict([newbie.user_input(), np.array([chest_list[i]])])
    print(recommend_predict)
    recommend_list.append(recommend_predict[0,0])

a=np.array(recommend_list).argsort()
print(a)