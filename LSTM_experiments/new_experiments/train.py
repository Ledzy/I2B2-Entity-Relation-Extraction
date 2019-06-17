from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Dropout, Embedding, Softmax, CuDNNGRU
from keras.layers import Bidirectional, concatenate, RepeatVector, Dot, Activation, merge, Reshape, Add
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import f1, get_train, clean_str, load_glove
import tensorflow as tf
import os #for setting GPU
import time #for formatting log name

from preprocess import * #run preprocess file

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# hyperparameters
LR = 1e-4
EPOCHS = 50

# define layer obejcts for achieving attention
repeat_vec = RepeatVector(maxlen)
densor = Dense(1,activation='relu') # repeat_vec & densor are used for one_step_attention
activator = Activation('softmax',name='attention_weights')
dotor = Dot(axes=1,name='context')
densor2 = Dense(1,activation='relu')

def one_step_attention(s_prev,a):
    """
    Note: This attention method is not applicable for the task, since the dataset is small. Instead,
    using one_step_attention_v2.

    calculate the weight of each word for the given input, using second LSTM's previous
    hidden state and the output of the first LSTM (see attention_lstm_old.png)

    parameters:
        s_prev: the hidden state of the second LSTM
        a: the output of the first LSTM
    """
    s_prev = repeat_vec(s_prev)
    concat = concatenate([s_prev,a],axis=-1)
    concat = Activation(activation='tanh')(concat)
    e = densor(concat)
    alphas = activator(e)
    context = dotor([alphas,a])
    
    return context


def one_step_attention_v2(a):
    """
    use only the previous state to form attention weights, see attention_lstm.png

    parameters:
        a: the output of the first LSTM
    """
    e = densor2(a)
    alphas = activator(e)
    context = dotor([alphas,a])
    
    return context


def model():
    
    sequences = Input(shape=(maxlen,),name='sequences')
    info_matrix = Input(shape=(maxlen,3),name='info_matrix')
    
    embedding = Embedding(max_features,emb_size,weights=[emb_matrix],trainable=False,name='embedding')(sequences)
    X = concatenate([embedding,info_matrix],axis=2,name='concat')
    a = Bidirectional(CuDNNLSTM(64,return_sequences=True))(X)
    
    context = one_step_attention_v2(a)
    context = Activation(activation='tanh')(context)
    
    output = Dense(tags.shape[1],activation='softmax')(context)
    output = Reshape((tags.shape[1],))(output)
    
    model = Model(inputs=[sequences,info_matrix],outputs=output)
    return model


# run the model
def run_model(record=True,validation_split=0.15,epochs=50,lr=1e-3):
    deep_model = model()

    opt = Adam(lr=1e-3)
    deep_model.compile(opt,loss='categorical_crossentropy',metrics=[f1,'accuracy'])
    
    if record == True:
        deep_model.fit([sequences,info_matrix],tags,epochs=epochs,validation_split=validation_split,callbacks=[tensorboard])
    else: deep_model.fit([sequences,info_matrix],tags,epochs=epochs,validation_split=validation_split)
        
    return deep_model
        
# adjust the NAME according to your needs
NAME = "Attention_v2-Simplify-Para-BiLSTM-Freeze-Embedding-Add-Pos-Preprocessing{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

deep_model = run_model(lr=LR,epochs=EPOCHS)