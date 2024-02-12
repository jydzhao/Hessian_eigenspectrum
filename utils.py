import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def accuracy(out, yb):
    '''
    calculates the accurac based on the predicted outputs and true labels y

    out: predicted output
    yb: true labels y
    '''
    preds = torch.sign(out)
#     yb = torch.argmax(yb)
   
    return (preds == yb).float().mean()


def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, bs):
    '''
    create Torch.Dataloaders based on given train-, test- and validation-data

    x_train: training input
    y_train: training labels
    x_val: validation input
    y_val: validation labels
    x_test: test input
    y_test: test labels
    bs: Batch size
    '''

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    valid_ds = TensorDataset(x_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=bs)
    
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)

    return train_dl, valid_dl, test_dl

def save_run(filename, x_train):
    '''
    helper function to save training data used

    filename: file name, where data is to be saved
    x_train: training data used
    '''
    with open(filename, 'wb') as f:
        np.save(f, x_train)
      
def load_run(filename):
    '''
    helper function to load training data used

    filename: file name, where training data is saved
    '''
    # to load data again:
    with open(filename, 'rb') as f:
        x_train = np.load(f)      
    
    return x_train