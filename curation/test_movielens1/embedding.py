import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Lambda, Embedding, Flatten, Multiply, RepeatVector, concatenate, Dense, Input
from tensorflow.keras.models import Model

def embedding_model(num_users, num_items, layers=[16,32,64,32,16,8]):
    
    num_layer = len(layers)

    MF_embedding_item = Embedding(input_dim = num_items+1, output_dim = num_users)
    MLP_embedding_item = Embedding(input_dim = num_items+1, output_dim = num_users) 
    
    user_input = Input(shape = (num_users,), dtype='float32', name='user_unput')
    item_input = Input(shape = (num_items,), dtype='int32', name='item_input')
    
    
    #MF part
    mf_user_latent = (user_input)
    mf_item_latent = (MF_embedding_item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    print(tf.shape(mf_vector))

    #MLP part
    mlp_user_latent = RepeatVector(num_items)(user_input)
    mlp_item_latent = (MLP_embedding_item(item_input))
    mlp_vector = concatenate([mlp_user_latent,mlp_item_latent],axis =2)
    print(tf.shape(mlp_vector))

    for idx in range(0, num_layer):
        layer = Dense(layers[idx], activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
    

    predict_vector = concatenate([mf_vector, mlp_vector])
    prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    #MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    #MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    #MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
    
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis = 0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('predictioin').set_weights([0.5*new_weights, 0.5*new_b])
    return model
