import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input
from tensorflow.keras.models import Model

def embedding_model(num_users, num_items, layers=[32,64,32,16,8,4]):
    
    num_layer = len(layers)

    MF_embedding_item = Embedding(input_dim = num_items+1, output_dim = 11)
    MLP_embedding_item = Embedding(input_dim = num_items+1, output_dim = 11) 
    
    user_input = Input(shape = (11,), dtype='float32', name='user_unput')
    item_input = Input(shape = (1,), dtype='int32', name='item_input')
    
    
    #MF part
    mf_user_latent = (user_input)
    mf_item_latent = Flatten()(MF_embedding_item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    mf_vector = (mf_vector)

    print(mf_vector)
    #MLP part
    mlp_user_latent = (user_input)
    mlp_item_latent = Flatten()(MLP_embedding_item(item_input))
    mlp_vector = concatenate([mlp_user_latent,mlp_item_latent],axis = 1)

    for idx in range(0, num_layer):
        layer = Dense(layers[idx], activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
    

    predict_vector = concatenate([(mf_vector), mlp_vector])
    prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model