# coding=utf-8

from keras.layers import Dense, Input, LSTM ,Dropout, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import concatenate
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from utils import get_tag_to_index, get_max_length, sentences_to_indices, load_word_map, prob_test_matrix
from embedding import pretrained_embedding_layer
from gensim.models.keyedvectors import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import random
 



#Define the model
def classifier(input_shape, input_shape2, word_to_vec_map, word_to_index):
    
    sentence_indices = Input(input_shape, dtype = 'int32')

    prob_test_oht = Input(input_shape2, dtype = 'float32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    #concatenate the embedding with prob_test_oht
    embeddings = concatenate([embeddings,prob_test_oht], axis=-1)

    #propogate the data through layers
    X = LSTM(128, return_sequences = True)(embeddings)
    X = Dropout(0.4)(X)
    X = LSTM(128, return_sequences = False)(X)
    X = Dropout(0.4)(X)
    X = Dense(6)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=[sentence_indices,prob_test_oht], outputs = X)

    return model


#load the file and embedding
print('Reading the file...')
with open('training file.txt', 'r') as f:
    data = f.readlines()
print('File closed.\n')

print('Loading the word-embedding...')
word_to_vec_map = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
print('Word-embedding loaded.\n')


word_to_index, index_to_word = load_word_map(data)
tag_to_index_map = get_tag_to_index(data)

Y_tags = data[3::5]
Y_indices = [tag_to_index_map[tag] for tag in Y_tags]

# Y_indices = [index for index in tag_to_index_map[Y_tags]]

X = np.array(data[0::5])
# X = np.random.shuffle(X)

max_len = get_max_length(X)


prob_test_matrix = prob_test_matrix(data,max_len)


#format the input of the model
X_train_indices = sentences_to_indices(X,word_to_index,max_len)
Y_train = to_categorical(Y_indices)



#balance the training set
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_sample(X_train_indices, Y_train)
prob_test_matrix,_ = ros.fit_sample(prob_test_matrix, Y_train)


#expand the dimension
prob_test_matrix_resampled = []
prob_test_dim = 32
for i in range(prob_test_dim):
    prob_test_matrix_resampled.append(prob_test_matrix)

#format the dimension
prob_test_matrix_resampled = np.transpose(prob_test_matrix_resampled,(1,2,0))


#shuffle the dataset 
index = [i for i in range(len(prob_test_matrix_resampled))]
random.shuffle(index)
prob_test_matrix_resampled = np.array([prob_test_matrix_resampled[i] for i in index])
X_resampled = np.array([X_resampled[i] for i in index])
Y_resampled = np.array([Y_resampled[i] for i in index])



#build, compile the model and start training
print('Constructing the model...')
model = classifier((max_len,),(max_len,32),word_to_vec_map,word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_resampled,prob_test_matrix_resampled], Y_resampled, epochs=5, batch_size=500,validation_split=0.2, shuffle=True)


#save the model
model.save('pre_shuffled.h5')