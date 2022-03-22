from user_info import input_bodyInfo
from user_info import person
from load_dataset import load_numpy_dataset
import numpy as np
from embedding import embedding_model
import tensorflow as tf
from tensorflow.keras.utils import plot_model


# name, height, weight, sex, age = input_bodyInfo()
# newbie = person(name, height, weight, sex, age)
# newbie.check()

user_dataset = load_numpy_dataset('user')
chest_list, chest_dataset = load_numpy_dataset('chest')
lower_list, lower_dataset = load_numpy_dataset('lower')
abdominals_list, abdominals_dataset = load_numpy_dataset('abdominals')
num_chest_dataset = len(chest_dataset[0])
num_user_dataset = len(user_dataset[0])

print(chest_list)

model = embedding_model(num_user_dataset,num_chest_dataset)
model.summary()
plot_model(model, show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy')
history = model.fit([user_dataset, chest_dataset])
