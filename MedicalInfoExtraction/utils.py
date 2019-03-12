import numpy as np
from keras.utils import to_categorical

#get the function of word_to_index and index_to_word. 
def load_word_map(data):
    #temp solution
    with open('training file.txt', 'r') as f:
        data = f.read()

    word_to_index = {}
    index_to_word = {}
    index = 0
    for word in set(data.split()):
        word_to_index[word] = index
        index += 1
    
    for word, index in word_to_index.items():
        index_to_word[index] = word

    return word_to_index, index_to_word


#get the length of the longest sequence in the input sequence
def get_max_length(input_sequence):
    max_len = 0

    for sentence in input_sequence: 
        length = len(sentence.split())
        if length > max_len:
            max_len = length
    
    return max_len


#transfer the sentences into indices to get the input of embedding layer
def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = [w for w in X[i].split()]

        j = 0
        for word in sentence_words:
            X_indices[i,j] = word_to_index[word]
            j += 1
        
    return X_indices


#get 6 tags from the data
def get_tag_to_index(data):
    tag = set(data[3::5])
    tag_to_index = {}
    index = 0
    for i in tag:
        tag_to_index[i] = index
        index += 1
    return tag_to_index