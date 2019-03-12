# coding=utf-8

from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from utils import get_tag_to_index, get_max_length, sentences_to_indices, load_word_map
from embedding import pretrained_embedding_layer
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

 



#Build the model
def classifier(input_shape, word_to_vec_map, word_to_index):
    
    sentence_indices = Input(input_shape, dtype = 'int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    #propogate the data through layers
    X = LSTM(128, return_sequences = True)(embeddings)
    X = Dropout(0.3)(X)
    X = LSTM(128, return_sequences = False)(X)
    X = Dropout(0.3)(X)
    X = Dense(6)(X)
    X = Activation('softmax')(X)

    model = Model(inputs = sentence_indices, outputs = X)

    return model


with open('training file.txt', 'r') as f:
    data = f.readlines()



word_to_vec_map = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
word_to_index, index_to_word = load_word_map(data)
tag_to_index_map = get_tag_to_index(data)

Y_tags = data[3::5]
Y_indices = [tag_to_index_map[tag] for tag in Y_tags]
Y_indices.pop()
# Y_indices = [index for index in tag_to_index_map[Y_tags]]

X = np.array(data[0::5])

max_len = get_max_length(X)

X_train_indices = sentences_to_indices(X,word_to_index,max_len)
Y_train = to_categorical(Y_indices)

model = classifier((max_len,),word_to_vec_map,word_to_index)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_indices,Y_train, epochs=3, batch_size=100,validation_split=0.1, shuffle=True)
