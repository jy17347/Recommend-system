import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input, Dropout
from tensorflow.keras.models import Model

def embedding_model(layers=[32,64,32,8]):
    
    num_layer = len(layers)

    MF_embedding_rent = Embedding(input_dim = 9, output_dim = 2)
    MLP_embedding_rent = Embedding(input_dim = 9, output_dim = 2)

    MF_embedding_body = Embedding(input_dim = 8, output_dim = 2)
    MLP_embedding_body = Embedding(input_dim = 8, output_dim = 2)

    MF_embedding_cat = Embedding(input_dim = 69, output_dim = 4)
    MLP_embedding_cat = Embedding(input_dim = 69, output_dim = 4)

    MF_embedding_id = Embedding(input_dim = 6000, output_dim = 4)
    MLP_embedding_id = Embedding(input_dim = 6000, output_dim = 4)
    
    user_input = Input(shape = (5,), dtype='float32', name='user_input')
    rent_embedding_input = Input(shape = (1,), dtype='float32', name='rent_input')
    body_embedding_input = Input(shape = (1,), dtype='float32', name='body_input')

    item_input = Input(shape = (1,), dtype='float32', name='item_input')
    id_embedding_input = Input(shape = (1,), dtype='float32', name='id_input')
    cat_embedding_input = Input(shape = (1,), dtype='float32', name='cat_input')
    
    
    #MF part
    mf_rent_latent = (MF_embedding_rent(rent_embedding_input))
    mf_rent_latent = Flatten()(mf_rent_latent)
    mf_body_latent = (MF_embedding_body(body_embedding_input))
    mf_body_latent = Flatten()(mf_body_latent)
    mf_user_latent = concatenate([user_input, mf_rent_latent, mf_body_latent])

    mf_cat_latent = (MF_embedding_cat(cat_embedding_input))
    mf_cat_latent = Flatten()(mf_cat_latent)
    mf_id_latent = (MF_embedding_id(id_embedding_input))
    mf_id_latent = Flatten()(mf_id_latent)
    mf_item_latent = concatenate([item_input, mf_cat_latent, mf_id_latent])
    
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])


    #MLP part
    mlp_rent_latent = (MLP_embedding_rent(rent_embedding_input))
    mlp_rent_latent = Flatten()(mlp_rent_latent)
    mlp_body_latent = (MLP_embedding_body(body_embedding_input))
    mlp_body_latent = Flatten()(mlp_body_latent)
    mlp_user_latent = concatenate([user_input, mlp_rent_latent, mlp_body_latent])
    
    mlp_cat_latent = (MLP_embedding_cat(cat_embedding_input))
    mlp_cat_latent = Flatten()(mlp_cat_latent)
    mlp_id_latent = (MLP_embedding_id(id_embedding_input))
    mlp_id_latent = Flatten()(mlp_id_latent)
    mlp_item_latent = concatenate([item_input, mlp_cat_latent, mlp_id_latent])

    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])

    for idx in range(0, num_layer):
        mlp_vector = Dense(layers[idx], activation='relu', name="layer%d" %idx)(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)
    

    predict_vector = concatenate([mf_vector, mlp_vector])
    prediction = Dense(3, activation='softmax', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, rent_embedding_input, body_embedding_input, id_embedding_input, cat_embedding_input, item_input], outputs=prediction)
    
    return model
