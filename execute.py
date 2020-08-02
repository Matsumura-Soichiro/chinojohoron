import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from src.net import Net, Unpool
from src.gen_data import gen_data, load_data
import s3fs

PATH = "data"
DATA_SIZE = 4200
TRAIN_DATA_SIZE = 3780
FEATURE_DIM = 9
HIDDEN_DIM = 7
EMBED_DIM = 5
CLASS_NUM = 12


def renumber_atom(x):
    x = np.where(x == 5, 0, x)
    x = np.where(x == 6, 1, x)
    x = np.where(x == 7, 2, x)
    x = np.where(x == 8, 3, x)
    x = np.where(x == 9, 4, x)
    x = np.where(x == 14, 5, x)
    x = np.where(x == 15, 6, x)
    x = np.where(x == 16, 7, x)
    x = np.where(x == 17, 8, x)
    x = np.where(x == 34, 9, x)
    x = np.where(x == 35, 10, x)
    x = np.where(x == 53, 11, x)
    return x


gen = load_data(PATH, DATA_SIZE)
data = gen.load_df()
train = data.iloc[:TRAIN_DATA_SIZE]
test = data.iloc[TRAIN_DATA_SIZE:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(FEATURE_DIM, HIDDEN_DIM, EMBED_DIM, CLASS_NUM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
Loss = nn.CrossEntropyLoss()


def train(epoch):
    model.train()

    loss_all = 0
    for i in range(1):
        x = torch.tensor(train.iloc[i]['feature'].values, device=device)
        g = torch.LongTensor(train.iloc[i]['inc'].values, device=device)
        y = torch.tensor(
            renumber_atom(train.iloc[i]['atoms'].values), device=device, dtype=torch.long
        )

        optimizer.zero_grad()
        output = model(x.float(), g)
        loss = Loss(output, y)
        loss.backward()
        pred = output.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        print("Loss: {:.5f}, Acc: {:.5f}".format(loss.item(), correct / x.shape[0]))
        optimizer.step()
    torch.save(model.state_dict(), "best_dgunet.pkl")
    return loss_all


for epoch in range(0, 10):
    loss = train(epoch)

train = []
for i in range(TRAIN_DATA_SIZE):
    x = torch.tensor(train.iloc[i]['feature'].values, device=device)
    g = torch.LongTensor(train.iloc[i]['inc'].values, device=device)
    embed = model.embed(x.float(), g)
    embed = torch.squeeze(embed)
    embed = embed.to("cpu").detach().numpy().copy()
    train.append(embed)
train = pd.DataFrame(np.array(train))
path2 = PATH + "/results/train.txt"
train.to_csv(path2, sep=" ", index=False, header=False)


test = []
for i in range(TRAIN_DATA_SIZE, DATA_SIZE):
    x = torch.tensor(test.iloc[i]['feature'].values, device=device)
    g = torch.LongTensor(test.iloc[i]['inc'].values, device=device)
    embed = model.embed(x.float(), g)
    embed = torch.squeeze(embed)
    embed = embed.to("cpu").detach().numpy().copy()
    test.append(embed)
test = pd.DataFrame(np.array(test))
path2 = PATH + "/results/test.txt"
test.to_csv(path2, sep=" ", index=False, header=False)
