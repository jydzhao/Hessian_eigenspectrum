import torch
from torch.utils.data import TensorDataset, DataLoader

def accuracy(out, yb):
    preds = torch.sign(out)
#     yb = torch.argmax(yb)
   
    return (preds == yb).float().mean()


def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, bs):

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    valid_ds = TensorDataset(x_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)

    return train_dl, valid_dl, test_dl