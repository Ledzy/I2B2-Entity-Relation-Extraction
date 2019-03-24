import numpy as np
from keras.utils import to_categorical
import re

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
            try:
                X_indices[i,j] = word_to_index[word]
                j += 1
            except Exception as e:
                print(e)
                print(word, end='')
        
    return X_indices


#get 6 tags from the data
def load_tag(data):
    tag = set(data[3::5])
    tag_to_index = {}
    index_to_tag = {}
    index = 0
    for i in tag:
        tag_to_index[i] = index
        index += 1

    for index, tag in tag_to_index.items():
        index_to_tag[index] = tag

    return tag_to_index,index_to_tag

'''get the problem/test information from data,
    and build a matrix such that for word tagged
    as problem, the relative cell would be 1. For 
    test words, the relative cell would be -1
'''
def prob_test_matrix(data,max_len):
    test_info = data[1::5]
    prob_info = data[2::5]
    test_pos = []
    prob_pos = []

    for key in test_info:
        pos = [int(i) for i in re.findall('\d+',key)]
        test_pos.append(pos)

    for key in prob_info:
        pos = [int(i) for i in re.findall('\d+',key)]
        prob_pos.append(pos)

    # test_pos_oht = to_categorical(test_pos)
    # prob_pos_oht = -to_categorical(prob_pos)

    prob_test_matrix = np.zeros([len(test_info),max_len])

    for i in range(len(test_info)):

        prob_test_matrix[i,prob_pos[i][0]:prob_pos[i][1]+1] = 1
        prob_test_matrix[i,test_pos[i][0]:test_pos[i][1]+1] = -1

    return prob_test_matrix