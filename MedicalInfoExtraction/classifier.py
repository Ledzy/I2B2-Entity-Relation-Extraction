# coding=utf-8

from keras.layers import Dense, Input, LSTM ,Dropout, Activation, Bidirectional
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import concatenate
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from utils import *
from embedding import pretrained_embedding_layer
from gensim.models.keyedvectors import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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
    X = Bidirectional(LSTM(128, return_sequences = True))(embeddings)
    X = Dropout(0.4)(X)
    X = Bidirectional(LSTM(128, return_sequences = False))(X)
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
tag_to_index_map, index_to_tag_map = load_tag(data)

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
# ros = RandomOverSampler(random_state=0)
ros = RandomUnderSampler(replacement=True, random_state=0)    #Reduce the size of largest tags
X_resampled, Y_resampled = ros.fit_sample(X_train_indices, Y_train)
prob_test_matrix,_ = ros.fit_sample(prob_test_matrix, Y_train)


#expand the dimension
pt_matrix = []
prob_test_dim = 32
for i in range(prob_test_dim):
    pt_matrix.append(prob_test_matrix)

#format the dimension
pt_matrix = np.transpose(pt_matrix,(1,2,0))


#shuffle the dataset 
index = [i for i in range(len(pt_matrix))]
random.shuffle(index)
pt_matrix = np.array([pt_matrix[i] for i in index])
X_resampled = np.array([X_resampled[i] for i in index])
Y_resampled = np.array([Y_resampled[i] for i in index])

#split the data into five parts for cross validation, then train the model
data_split = 5
train_log = []
for i in range(data_split):
    start = int(i/5*len(X_resampled))
    end = int((i+1)/5*len(X_resampled))

    X_resampled_test = X_resampled[start:end]
    Y_resampled_test = Y_resampled[start:end]
    pt_matrix_test = pt_matrix[start:end]

    X_resampled_train = np.append(X_resampled[0:start],X_resampled[end:],axis=0)
    Y_resampled_train = np.append(Y_resampled[0:start],Y_resampled[end:],axis=0)
    pt_matrix_train = np.append(pt_matrix[0:start],pt_matrix[end:],axis=0)

    print('Constructing the model, split ',i)
    model = classifier((max_len,),(max_len,32),word_to_vec_map,word_to_index)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    History = model.fit([X_resampled_train,pt_matrix_train], Y_resampled_train, epochs=5, batch_size=50,validation_data=([X_resampled_test,pt_matrix_test],Y_resampled_test), shuffle=True)

    history = History.history
    acc,loss,val_acc,val_loss = history['acc'][-1], history['loss'][-1], history['val_acc'][-1], history['val_loss'][-1]
    train_log.append([acc,loss,val_acc,val_loss])


#print the performance of the model
model = classifier((max_len,),(max_len,32),word_to_vec_map,word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_resampled,pt_matrix],Y_resampled,epochs=30,validation_split=0.2,batch_size=100,shuffle=True)


#save the model
model.save('add_bidirect.h5')