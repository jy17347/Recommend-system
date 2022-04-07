import time, os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import load_dataset
from get_dataset import get_trainset
from get_dataset import scaler_user
from embedding import embedding_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


created_time = int(time.time()) 
user_dataset = load_dataset('user')
chest_list, chest_dataset = load_dataset('chest')
# lower_list, lower_dataset = load_numpy_dataset('lower')
# abdominals_list, abdominals_dataset = load_numpy_dataset('abdominals')
user_dataset = scaler_user(user_dataset)

num_chest_dataset = len(chest_dataset[0])
num_user_dataset = len(user_dataset[0])

user_train_input, chest_train_input, chest_label = get_trainset(user_dataset, chest_list, chest_dataset)
print(np.shape(user_train_input),np.shape(chest_train_input),chest_label)

model = embedding_model(num_user_dataset,num_chest_dataset)
model.summary()
plot_model(model, show_shapes=True)

x_train, x_val, y_train, y_val, label_train, label_val = train_test_split(user_train_input, chest_train_input, chest_label, test_size=0.2, random_state=2022)
loss = input("loss: ")
model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.Precision(thresholds=None, top_k=4, class_id=None, name=None, dtype=None)])
history = model.fit([x_train, y_train], label_train, epochs=50, validation_data=([x_val, y_val], label_val))

os.mkdir(f"model/{loss}_{created_time}")
model.save(f'model/{loss}_{created_time}/chest_model.h5')


pd.Series(history.history['loss']).plot(logy=True)
pd.Series(history.history['val_loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.savefig(f"model/{loss}_{created_time}/train_error.png")
plt.show()
print(label_train)