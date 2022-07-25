import numpy as np
import pandas as pd
import tensorflow as tf
import os
from get_dataset import load_dataset
from get_dataset import scaler_user
from tensorflow.keras.models import load_model
exer_name = pd.read_csv('C:/project/kwix/curation/dataset/exercise_list.csv')
model_name = os.listdir('C:/project/kwix/curation/neural_precisionK/model/')[-1]
model = load_model(f'model/{model_name}/chest_model.h5')
chest_list, temp = load_dataset('chest')
print(model.layers[5].get_weights)
