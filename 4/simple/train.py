import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.utils.data as data
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import math
import torch.backends.cudnn as cudnn

from models import CNN, RNN, BiLSTM, MLP

drive_dir = "/content/drive/My Drive/app/"
drive_dir = ""
net_type = "RNN"
batch_size = 1024
padding = 64
droprate = 0
regularization = 0.05
epochs = 100
seed = 0
num_classes = 5

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set up main device and scale batch size
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Model
print("Building net...")
if net_type == "CNN":
    net = CNN(droprate)
if net_type == "RNN":
    net = RNN(droprate, padding)
if net_type == "BiLSTM":
    net = BiLSTM(droprate, padding)
if net_type == "MLP":
    net = MLP(droprate, padding)
if device == "cuda":
    net.to(device)
cudnn.benchmark = True

optimizer = optim.Adam(net.parameters(), weight_decay=regularization)


print("Loading training data...")
labelSet = np.load(drive_dir + "train-labels.npy")
weight = [0.3, 0.3, 0.15, 0.15, 0.1]
print("Weight:", weight)
loss_func = nn.CrossEntropyLoss(reduction="sum", weight=torch.from_numpy(np.array(weight)).float()).to(device)

labelSet = torch.LongTensor(labelSet)
print("Traing label shape:", labelSet.shape)

inputSet = np.load(drive_dir + "train-50d.npy")
inputSet = torch.from_numpy(inputSet)
inputSet = torch.unsqueeze(inputSet, 1)
inputSet = inputSet.type(torch.FloatTensor)
print("Traing input shape:", inputSet.shape)

length = len(labelSet)
len_train = length // 10 * 9
idxes = np.arange(length)
np.random.seed(1234)
np.random.shuffle(idxes)

trainData, trainLabel = inputSet[idxes[:len_train]], labelSet[idxes[:len_train]]
valData, valLabel = inputSet[idxes[len_train:]], labelSet[idxes[len_train:]]

trainset = data.TensorDataset(trainData, trainLabel)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

valset = data.TensorDataset(valData, valLabel)
valloader = data.DataLoader(valset, batch_size=batch_size, shuffle=False)

print("Training data loaded")

trainlosses = []
testlosses = []


def cal_test(label, pred):
    acc = np.mean(label == pred)
    mae = np.mean(np.abs(label - pred))
    rmse = math.sqrt(np.mean(np.square(label - pred)))
    print("Acc:", acc, "MAE:", mae, "RMSE:", rmse)
    return acc, mae, rmse


best_acc = 0
best_mae = 0
best_rmse = 0
for epoch in range(epochs):
    print("Epoch", epoch)
    train_loss = 0
    net.train()
    for x, y in trainloader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        predict = net(x)
        loss = loss_func(predict, y)
        loss.backward()
        train_loss += float(loss.data)
        optimizer.step()
    train_loss /= len(trainloader.dataset)
    print("Train loss:", train_loss)
    trainlosses.append(train_loss)
    net.eval()
    preds = []
    true_y = []
    with torch.no_grad():
        for x, y in valloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = net(x)
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            preds.extend(y_pred)
            true_y.extend(y.squeeze().cpu().numpy().tolist())
        acc, mae, rmse = cal_test(np.array(true_y), np.array(preds))
        if acc > best_acc:
            best_acc = acc
            best_mae = mae
            best_rmse = rmse
    torch.cuda.empty_cache()

print(trainlosses)
