import time, os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import load_dataset
from get_dataset import get_trainset
from get_dataset import scaler_user
from embedding import embedding_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

created_time = int(time.time()) 
user, item, label = load_dataset()

body = user[:,6].astype(np.float32)
rent = user[:,5].astype(np.float32)
user = user[:,0:5].astype(np.float32)
id = item[:,0].astype(np.float32)
cat = item[:,1].astype(np.float32)
item = item[:,2].astype(np.float32)

model = embedding_model()
model.summary()
plot_model(model, show_shapes=True)


user_train, user_val, user_embedding_train, user_embedding_val, item_train, item_val, cat_train, cat_val, id_train, id_val, label_train, label_val = train_test_split(user, rent, body, item, cat, id, label, test_size=0.1, random_state=2022)

print(np.shape(user_train))
print(np.shape(user_embedding_train))
print(np.shape(item_train))
print(np.shape(cat_train))
print(np.shape(id_train))
print(np.shape(label_train[:,1]))


loss = 'binary_crossentropy'
model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.AUC()])
history = model.fit([user_train, user_embedding_train, id_train, cat_train, item_train], label_train[:,1], epochs=50, batch_size=32,validation_data=([user_val, user_embedding_val, id_val, cat_val, item_val], label_val[:,1]))

os.mkdir(f"model/{loss}_{created_time}")
model.save(f'model/{loss}_{created_time}/chest_model.h5')

pd.Series(history.history['loss']).plot(logy=True)
pd.Series(history.history['val_loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("train Error")
plt.savefig(f"model/{loss}_{created_time}/train_error.png")
plt.show()