import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input, Dropout
from tensorflow.keras.models import Model

def embedding_model(num_users, num_items, layers=[]):
    

    MF_embedding_user = Embedding(input_dim = 14000, output_dim = 8)
    MF_embedding_item = Embedding(input_dim = 16, output_dim = 8)

    
    user_input = Input(shape = (1,), dtype='float32', name='user_unput')
    item_input = Input(shape = (1,), dtype='int32', name='item_input')
    
    
    #MF part
    mf_user_latent = Flatten()(MF_embedding_user(user_input))
    mf_item_latent = Flatten()(MF_embedding_item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    prediction = Dense(1, activation='sigmoid', name='prediction')(mf_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model