from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer

drive_dir = "/content/drive/My Drive/app/"
pretrained = "bert-large-uncased"
SEQ_LEN = 128

with open(drive_dir + "test.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]

x = []
for line in lines:
    words = line.split("\t")
    x.append(words[-2] + " " + words[-1])

tokenizer = BertTokenizer.from_pretrained(pretrained)

input_ids, attention_masks, input_types = [], [], []

for line in x:
    assert line != ""
    sample = tokenizer(line, max_length=SEQ_LEN, padding="max_length", truncation=True)
    input_ids.append(sample["input_ids"])
    attention_masks.append(sample["attention_mask"])
    input_types.append(sample["token_type_ids"])

input_ids, attention_masks, input_types = np.array(input_ids), np.array(attention_masks), np.array(input_types)

np.save(drive_dir + "test_ids.npy", input_ids)
np.save(drive_dir + "test_masks.npy", attention_masks)
np.save(drive_dir + "test_types.npy", input_types)

input_ids = np.load(drive_dir + "test_ids.npy")
attention_masks = np.load(drive_dir + "test_masks.npy")
input_types = np.load(drive_dir + "test_types.npy")

BATCH_SIZE = 32

test_data = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(attention_masks), torch.LongTensor(input_types))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


class Bert(nn.Module):
    def __init__(self, classes=5):
        super(Bert, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained)
        self.bert = BertModel.from_pretrained(pretrained)
        self.fc = nn.Linear(self.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        logit = self.fc(out_pool)
        return logit


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Bert().to(DEVICE)


def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


model.load_state_dict(torch.load(drive_dir + "best_bert_model.pth"))
predicts = predict(model, test_loader, DEVICE)
with open(drive_dir + "predict.csv", "w", encoding="utf-8") as f:
    lines = ["Id,Predicted\n"]
    for i in range(1, 10000 + 1):
        lines.append("{},{}\n".format(i, float(predicts[i - 1] + 1)))
    f.writelines(lines)