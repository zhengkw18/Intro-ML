import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

drive_dir = "/content/drive/My Drive/app/"
pretrained = "bert-large-uncased"
SEQ_LEN = 128

with open(drive_dir + "train.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]

x, y = [], []
for line in lines:
    words = line.split("\t")
    x.append(words[-2] + " " + words[-1])
    y.append(int(float(words[0])))

tokenizer = BertTokenizer.from_pretrained(pretrained)

input_ids, attention_masks, input_types = [], [], []

for line in x:
    assert line != ""
    sample = tokenizer(line, max_length=SEQ_LEN, padding="max_length", truncation=True)
    input_ids.append(sample["input_ids"])
    attention_masks.append(sample["attention_mask"])
    input_types.append(sample["token_type_ids"])

input_ids, attention_masks, input_types = np.array(input_ids), np.array(attention_masks), np.array(input_types)

y = np.array(y)
print(input_ids.shape, attention_masks.shape, input_types.shape, y.shape)

np.save(drive_dir + "train_ids.npy", input_ids)
np.save(drive_dir + "train_masks.npy", attention_masks)
np.save(drive_dir + "train_types.npy", input_types)
np.save(drive_dir + "train_y.npy", y)

input_ids = np.load(drive_dir + "train_ids.npy")
attention_masks = np.load(drive_dir + "train_masks.npy")
input_types = np.load(drive_dir + "train_types.npy")
y = np.load(drive_dir + "train_y.npy") - 1

length = input_ids.shape[0]
len_train = length // 10 * 9
idxes = np.arange(length)
np.random.seed(1234)
np.random.shuffle(idxes)
id = 3
test_start = length // 10 * (9 - id)
test_end = test_start + length // 10
train_idx = [i for i in range(length) if i < test_start or i >= test_end]
test_idx = list(range(test_start, test_end))
input_ids_train, input_ids_val = input_ids[idxes[:len_train]], input_ids[idxes[len_train:]]
attention_masks_train, attention_masks_val = attention_masks[idxes[:len_train]], attention_masks[idxes[len_train:]]
input_types_train, input_types_val = input_types[idxes[:len_train]], input_types[idxes[len_train:]]
y_train, y_val = y[idxes[:len_train]], y[idxes[len_train:]]
weights = []
for i in range(5):
    weights.append(float(len(y) / (y == i).sum()))

BATCH_SIZE = 32

train_data = TensorDataset(torch.LongTensor(input_ids_train), torch.LongTensor(attention_masks_train), torch.LongTensor(input_types_train), torch.FloatTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(torch.LongTensor(input_ids_val), torch.LongTensor(attention_masks_val), torch.LongTensor(input_types_val), torch.FloatTensor(y_val))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained)
        self.bert = BertModel.from_pretrained(pretrained)
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        logit = self.fc(out_pool)
        return logit.reshape(-1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = Bert().to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=EPOCHS * len(train_loader))


def cal_test(label, pred):
    mae = np.mean(np.abs(label - pred))
    rmse = math.sqrt(np.mean(np.square(label - pred)))
    print("MAE:", mae, "RMSE:", rmse)


def evaluate(model, data_loader, device):
    val_true, val_pred = [], []
    with torch.no_grad():
        for ids, att, tpe, y in data_loader:
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred[y_pred < 0.0] = 0.0
            y_pred[y_pred > 4.0] = 4.0
            y_pred = y_pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    cal_test(np.array(val_true), np.array(val_pred))
    rmse = math.sqrt(np.mean(np.square(np.array(val_true) - np.array(val_pred))))
    return rmse


def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epoch, weights):
    best_rmse = 1e5
    weights = torch.tensor(weights).to(device)
    criterion = nn.MSELoss()
    for i in range(epoch):
        start = time.time()
        model.train()
        print("Epoch {}".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader) // 100) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}s".format(i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))

        model.eval()
        rmse = evaluate(model, valid_loader, device)
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), drive_dir + "best_bert_model.pth")

        print("current rmse is {:.4f}, best rmse is {:.4f}".format(rmse, best_rmse))


train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, DEVICE, EPOCHS, weights)