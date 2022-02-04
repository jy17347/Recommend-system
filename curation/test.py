import numpy as np
import time, os
from tensorflow.keras.models import load_model


action_list = np.load('action_list/action_list.npy')
model = load_model('models/curation_model.h5')
priority = list(range(1, 70, 5))
recommend_list = []
actions = []

for i in priority:
    
    input_data = [[i, 5, 1, 1],]
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    actions = action_list[i_pred]
    if actions in recommend_list:
        continue
    recommend_list.append(actions)
    
print(recommend_list)