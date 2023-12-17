import torch
import numpy as np 
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from sys import argv
from math import ceil, floor, sqrt
from stockbot import StockBot
from dataset import StockDataset
from hyperparameters import N_EPOCHS,LR, BATCH_SIZE, SAVE_PATH

def train(model, loss_fn, dl):
    pbar = tqdm(range(1,N_EPOCHS+1))
    lr = LR
    for i in pbar:
        optim = Adam(model.parameters(), lr=lr/sqrt(i+1))
        loss = train_one_epoch(model, loss_fn, optim, dl)
        pbar.set_description(f"epoch : {i}, loss : {loss}")


def train_one_epoch(model, loss_fn, optim, dl):
    running_loss, last_loss = [], 0
    for X,y in dl:
        model.train(True)
        X, y = X.float(), y.float()
        optim.zero_grad()
        y_pred = model(X.float())
        loss = loss_fn(y_pred, y)
        loss.backward()
        optim.step()
        last_loss = loss.item()
        running_loss.append(last_loss)
    return np.mean(running_loss)

def infer(model,test_dl, vrange):
    (_max, _min) = vrange
    print(vrange)
    for X,y in test_dl:
        X,y = X.float(), y.detach().numpy()
        y_pred = model(X)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred * (_max - _min) + _min
        y = y * (_max - _min) + _min

        print(y_pred.shape, y.shape)
        print("y",y[0],"y_pred", y_pred[0])
        print("MSE: ", np.mean((y_pred-y)**2))
        

def main():
    dataset = StockDataset("datasets/adani.csv")
    model = StockBot()

    l = len(dataset)
    train_set, test_set = random_split(dataset, [ceil(l*0.9), floor(l*0.1)])
    vrange = dataset.range

    def mdl (x,bs=BATCH_SIZE):
        return DataLoader(x, batch_size=bs, shuffle=True)
    train_dataloader, test_dataloader = mdl(train_set), mdl(test_set,1000)

    if len(argv) == 1:
        #model = torch.load("./models/hdfc.pt")
        loss_fn = nn.MSELoss(reduction="mean")
        train(model, loss_fn, train_dataloader)
        torch.save(model, SAVE_PATH)
        infer(model,test_dataloader,vrange)
    else:
        model = torch.load(f"./models/{argv[1]}.pt")
        infer(model,test_dataloader, vrange)
 
if __name__ == "__main__":
    main()   
