import numpy as np
import pandas as pd
import os, time
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

dataset_dir = 'C://project/dataset/curation'
data_list = os.listdir(dataset_dir)
os.makedirs('action_list', exist_ok=True)

data = pd.read_csv(dataset_dir + '/' + data_list[0], encoding = 'cp949')
data = data.to_numpy()

x_data = data[:,:-1]
labels = data[:,-1]


action_list = []
y_data = []
for action in labels:
    
    if (action in action_list) == False:
        action_list.append(action)
    
    y_data.append(action_list.index(action))

action_list = np.array(action_list)

np.save(os.path.join('action_list', 'action_list'), action_list)

y_data = to_categorical(y_data)

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2022)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


model = Sequential([
    Dense(16, activation='relu', input_shape = (4,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(action_list), activation='softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/curation_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)