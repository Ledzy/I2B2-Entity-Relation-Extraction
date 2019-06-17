"""
This file get preprocessed data
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import load_glove, clean_str, get_train
import pandas as pd
import numpy as np 
import re


# get input
train_df = get_train()
texts = train_df['text'].to_list()
tags = train_df['tag'].to_list()

# clean the text
train_df['text'] = train_df['text'].apply(clean_str)

# text2sequence
emb_size = 300
max_features = 6000
maxlen = 50

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences,maxlen=maxlen)

tokenizer_tag = Tokenizer()
tokenizer_tag.fit_on_texts(tags)
tags = tokenizer_tag.texts_to_sequences(tags)
tags = np.array(list((map(lambda x: x[0],tags))))
tags = to_categorical(tags)


# load embedding
emb_matrix = load_glove(word_index)

# Get test/problem/treatment matrix: info_matrix
# info_matrix: (m,3,maxlen), which uses one-hot to indicate the entity property of the token
targets = ['test_info','problem_info','treatment_info']
info_matrix = np.zeros((sequences.shape[0],3,maxlen))

for i,target in enumerate(targets):
    for k,j in train_df[target].str.extract('(\d+)\|(\d+)').iterrows():
        if not pd.isnull(j[0]): 
            info_matrix[k,i,int(j[0])-1:int(j[1])] = 1


# Shuffle the data
np.random.seed(2019)
index = np.random.permutation(len(sequences))

sequences = sequences[index]
tags = tags[index]
info_matrix = info_matrix[index].swapaxes(1,2)