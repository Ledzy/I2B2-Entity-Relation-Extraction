"""
This file is used for evaluating the model.
For this task, the classification report of sklearn was used, which contains
precision/recall/f1.
Feel free to add more metrics
"""

# let's firstly load the model and validation data
from __future__ import absolute_import, division, print_function
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pytorch_pretrained_bert import BertForSequenceClassification
from tqdm import tqdm_notebook
import torch
import os

FILE_PATH = "./bert_pytorch.bin"
VAL_X_PATH = "./val_X.npy"
VAL_Y_PATH = ".val_y.npy"

if __name__ == "__main__":

    # check if the train process has been completed and the file is put in the right place
    if os.path.exists(FILE_PATH):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=len(set(tags)))
        model.load_state_dict(torch.load("bert_pytorch.bin" ))
        model.to(device)
    else:
        print(f"\nmodel file not found, run train.py first to get the model file, and put it in {FILE_PATH}")
        raise FileNotFoundError

    if os.path.exists(VAL_X_PATH):
        val_X = np.load(VAL_X_PATH)
    else: 
        print(f"\nfile {VAL_X_PATH} not found, run train.py first to get the validation file, and put it in {VAL_X_PATH}")
        raise FileNotFoundError

    if os.path.exists(VAL_Y_PATH):
        val_y = np.load(VAL_Y_PATH)
    else:
        print(f"\nfile {VAL_Y_PATH} not found, run train.py first to get the validation file, and put it in {VAL_Y_PATH}")
        raise FileNotFoundError


    # freeze the model
    for param in model.parameters():
        param.requires_grad=False
    model.eval()

    valid_preds = []
    valid = torch.utils.data.TensorDataset(torch.tensor(val_X,dtype=torch.long))
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

    tk0 = tqdm_notebook(valid_loader)
    for i,(x_batch,)  in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        valid_preds.append(np.argmax(pred.cpu().numpy(),axis=1))
        
    valid_preds = np.concatenate(valid_preds,axis=0)

    print(classification_report(valid_preds,val_y))