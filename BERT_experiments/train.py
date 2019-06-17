from __future__ import absolute_import, division, print_function

import time
import gc
import re
import sys
import os
import warnings
import pandas as pd
import numpy as np 
import re
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, classification_report
from sklearn import model_selection

from tqdm import tqdm, tqdm_notebook
from IPython.core.interactiveshell import InteractiveShell
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam, BertModel
from utils import get_train, clean_str, convert_lines
from apex import amp

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings(action='once')
device=torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# hyperparameters
FILE_PATH = "training file.txt"
MAX_SEQUENCE_LENGTH = 75
TRAIN_SIZE = 6500
SEED = 666
EPOCHS = 5
LR=2e-5
BATCH_SIZE = 32
ACCUMULATION_STEPS=2 # how many steps it should backward propagate before optimization
OUTPUT_FILE_NAME = "bert_pytorch.bin"

# convert the origin data into a formatted pandas dataframe
train_df = get_train(FILE_PATH)
train_df['text'] = train_df['text'].apply(clean_str)

#convert tag to sequence, maybe there are more elegant way to do this
tags = train_df['tag'].to_list()
tokenizer_tag = Tokenizer()
tokenizer_tag.fit_on_texts(tags)
tags = tokenizer_tag.texts_to_sequences(tags)
tags = np.array(list((map(lambda x: x[0],tags))))


# convert text to bert format sequence
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sequences = convert_lines(train_df["text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)


#shuffle the data
np.random.seed(2019)
index = np.random.permutation(len(sequences))
sequences = sequences[index]
tags = tags[index]


# split the data into train/test
X = sequences[:TRAIN_SIZE]                
y = tags[:TRAIN_SIZE]
val_X = sequences[TRAIN_SIZE:]                
val_y = tags[TRAIN_SIZE:]
y = to_categorical(y-1)
val_y = to_categorical(val_y-1)

#due to the GPU memory limitation, just use 64 data to validate
#the complete validation process would be done after the training process is over
val_y = val_y[:64]
val_X = val_X[:64]
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

# Initialize the model
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
output_model_file = OUTPUT_FILE_NAME

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=len(set(tags)))
model.zero_grad()
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/BATCH_SIZE/ACCUMULATION_STEPS)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LR,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()

# train the model
tq = tqdm_notebook(range(EPOCHS))

for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()
    
    for i,(x_batch, y_batch) in tk0:
        torch.cuda.empty_cache()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % ACCUMULATION_STEPS == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        
        if i % 5 == 0:
            
            val_output = model(torch.tensor(val_X).to(device),attention_mask=(torch.tensor(val_X)>0).to(device), labels=None)
            val_pred = np.argmax(val_output.data.cpu(),axis=1)
            
            val_loss =  F.binary_cross_entropy_with_logits(val_output,torch.tensor(val_y).to(device))
            accuracy = torch.sum(torch.tensor(val_pred) == \
                                 torch.tensor(np.argmax(val_y,axis=1))).type(torch.FloatTensor) / torch.tensor(val_y).size(0)

            print('Step: ', i, '| train loss: %.4f' % lossf, '| test accuracy: %.2f' % accuracy,'| val loss: %2f' % val_loss.item())
    print(classification_report(np.argmax(val_y,axis=1),val_pred.numpy()))
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    
# save the model and validation data for evaluate
torch.save(model.state_dict(), output_model_file)
np.save("val_y.npy",tags[TRAIN_SIZE:])
np.save("val_X.npy",sequences[TRAIN_SIZE:])