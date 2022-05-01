import time
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import get_trainset
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model

created_time = int(time.time()) 
loss = input("loss function:")

model = load_model(f'model/binary_crossentropy_1649318650/chest_model.h5')

user_dataset = load_dataset('test/user')
user_dataset = scaler_user(user_dataset)
chest_list, chest_dataset = load_dataset('test/chest')
user_testset, chest_testset, chest_label = get_trainset(user_dataset, chest_list, chest_dataset)

binary_crossentropy, precision = model.evaluate([user_testset, chest_testset], chest_label, batch_size=1)
print("loss : ", binary_crossentropy)

pd.Series([binary_crossentropy]).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.savefig(f"model/binary_crossentropy_1649318650/test_error.png")
plt.show()
