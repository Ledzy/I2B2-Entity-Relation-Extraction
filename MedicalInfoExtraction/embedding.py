import numpy as np 
from keras.layers.embeddings import Embedding


#define a embedding layer which is initialized with pre-trained word-embedding
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    
    vocab_length = len(word_to_index) + 1  #adding 1 to fit the Embedding layer (keras requirement)
    emb_dim = word_to_vec_map['at'].shape[0]

    emb_matrix = np.zeros((vocab_length, emb_dim))

    success = 0
    fail = []
    for word, index in word_to_index.items():
        try:
            emb_matrix[index,:] = word_to_vec_map[word]
            success += 1

        except Exception as e:
            fail.append(word)
            emb_matrix[index,:] = -np.random.randn(emb_dim)/20


    embedding_layer = Embedding(vocab_length,emb_dim,trainable=True)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer



