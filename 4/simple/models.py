import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, droprate):
        super(CNN, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(1, 100, (3, 50)), nn.Conv2d(1, 100, (4, 50)), nn.Conv2d(1, 100, (5, 50))])
        self.drop = nn.Dropout(droprate)
        self.fc = nn.Linear(300, 5)

    def forward(self, x):
        a = []
        for i in range(3):
            a.append(self.conv[i](x))
            a[i] = a[i].view(a[i].size()[:-1])
            a[i] = F.max_pool1d(a[i], kernel_size=a[i].size()[-1:])
            a[i] = a[i].view(a[i].size()[:-1])
        x = self.drop(torch.cat((a[0], a[1], a[2]), 1))
        x = F.relu(self.fc(x))
        x = F.softmax(x, dim=1)
        return x


class RNN(nn.Module):
    def __init__(self, droprate, padding):
        super(RNN, self).__init__()
        self.encoder = nn.RNN(input_size=50, hidden_size=64, num_layers=2, bidirectional=False, dropout=droprate)
        self.decoder = nn.Linear(padding * 64, 5)

    def forward(self, x):
        self.encoder.flatten_parameters()
        x = torch.transpose(torch.squeeze(x, 1), 0, 1)
        x = self.encoder(x)[0].transpose(0, 1)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.decoder(x)
        x = F.softmax(x, dim=1)
        return x


class BiLSTM(nn.Module):
    def __init__(self, droprate, padding):
        super(BiLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, bidirectional=True, dropout=droprate)
        self.decoder = nn.Linear(2 * padding * 64, 5)

    def forward(self, x):
        self.encoder.flatten_parameters()
        x = torch.transpose(torch.squeeze(x, 1), 0, 1)
        x = self.encoder(x)[0].transpose(0, 1)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.decoder(x)
        x = F.softmax(x, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, droprate, padding):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(50 * padding, 512)
        self.drop = nn.Dropout(droprate)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
