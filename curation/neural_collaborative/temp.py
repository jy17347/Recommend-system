import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import argparse
from load_dataset import load_numpy_dataset


chest_dataset = load_numpy_dataset('chest')
num_chest_dataset = len(chest_dataset[0])



MF_embedding_item = Embedding(input_dim = num_chest_dataset, output_dim = 6)
MLP_embedding_item = Embedding(input_dim = num_chest_dataset, output_dim = 6) 
    
user_input = Input(shape = (6,), dtype='float32', name='user_unput')
item_input = Input(shape = (1,), dtype='int32', name='item_input')

mf_item_latent = (MF_embedding_item(item_input))
    
print(mf_item_latent.shape)