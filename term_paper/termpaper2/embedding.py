import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input, Dropout
from tensorflow.keras.models import Model

def embedding_model(layers=[32,64,32,16]):
    
    num_layer = len(layers)

    user_id = Input(shape=(1,), dtype='float32', name='user_id')
    age = Input(shape=(1,), dtype='float32', name='age')
    location_match = Input(shape=(1,), dtype='float32', name='match')

    isbn =Input(shape=(1,),dtype='float32', name='isbn')
    year = Input(shape=(1,), dtype='float32', name='year')
    lan = Input(shape=(1,), dtype='float32', name='lan')
    category = Input(shape=(1,), dtype='float32', name='category')
    
    
    user_id_embedding = Embedding(input_dim=86936+1, output_dim=9)
    isbn_embedding = Embedding(input_dim=241370+1, output_dim=6)
    lan_embedding = Embedding(input_dim=5+1, output_dim=2)
    category_embedding = Embedding(input_dim=30+1, output_dim=2)
    
    user_id_embed = Flatten()(user_id_embedding(user_id))
    user_latent = concatenate([user_id_embed, age, location_match], axis=1)

    isbn_embed = Flatten()(isbn_embedding(isbn))
    lan_embed = Flatten()(lan_embedding(lan))
    category_embed = Flatten()(category_embedding(category))
    book_latent = concatenate([isbn_embed, year, lan_embed, category_embed], axis=1)

    mf_vector = Multiply()([user_latent, book_latent])
    mlp_vector = concatenate([user_latent, book_latent], axis=1)    
    
    for idx in range(0, num_layer):
        mlp_vector = Dense(layers[idx], activation='relu', name="layer%d" %idx)(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)

    
    predict_vector = concatenate([(mf_vector), mlp_vector])
    prediction = Dense(11, activation='softmax', name='prediction')(predict_vector)
    model = Model(inputs=[user_id, age, location_match, isbn, year, lan, category], outputs=prediction)
    
    return model



    # MF_embedding_item = Embedding(input_dim = num_items+1, output_dim = 6)
    # MLP_embedding_item = Embedding(input_dim = num_items+1, output_dim = 6) 
    
    # user_input = Input(shape = (6,), dtype='float32', name='user_unput')
    # item_input = Input(shape = (1,), dtype='float32', name='item_input')
    
    
    # #MF part
    # mf_user_latent = (user_input)
    # mf_item_latent = Flatten()(MF_embedding_item(item_input))
    # mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    # mf_vector = (mf_vector)

    # print(mf_vector)
    # #MLP part
    # mlp_user_latent = (user_input)
    # mlp_item_latent = Flatten()(MLP_embedding_item(item_input))
    # mlp_vector = concatenate([mlp_user_latent,mlp_item_latent],axis = 1)


    # for idx in range(0, num_layer):
    #     mlp_vector = Dense(layers[idx], activation='relu', name="layer%d" %idx)(mlp_vector)
    #     mlp_vector = Dropout(0.2)(mlp_vector)
        

    # predict_vector = concatenate([(mf_vector), mlp_vector])
    # prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
    # model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    # return model

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
