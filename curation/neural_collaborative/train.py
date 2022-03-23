import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import load_dataset
from get_dataset import get_trainset
from get_dataset import scaler_user
from embedding import embedding_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



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

model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.Precision()])
history = model.fit([user_train_input, chest_train_input],chest_label, epochs=300)
model.save('model/curation_model.h5')


pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.show()
