import pandas as pd
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

date_train = pd.read_csv('train.csv')

def normazil(text):
  text = re.sub(r'http\S+', '', text)
  return text.lower()
date_train['text'] = date_train['text'].apply(normazil)

date_new = []
for text in date_train['text'].tolist():
  text = text.split(' ')
  new_text = ""
  for word in text:
    if '@' not in word and '#' not in word:
      new_text += f'{word} '
  
  date_new.append(new_text)

date_train['new_text'] = date_new


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

after_tokenizer = tokenizer(list(date_train['new_text']), padding=True, truncation=True,return_tensors='pt')
date_labels = torch.tensor(date_train['target'].values)

datasets = TensorDataset(after_tokenizer['input_ids'], after_tokenizer["attention_mask"], date_labels)
dataloader = DataLoader(datasets, 16)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Навчання
for epoch in range(3):
    model.train()
    los_all = []
    for index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        los_all.append(loss.item())
        if (index + 1) % 10 == 0:
            print(
                f"Epoch [\n\n\n{epoch+1}/{3}]\
                        Batch {index+1}/{len(dataloader)} "
                f"Loss: {sum(los_all)/len(los_all)} "

            )
            los_all = []
            print(los_all)

date_test = pd.read_csv('test.csv')

def normazil(text):
  text = re.sub(r'http\S+', '', text)
  return text.lower()
date_test['text'] = date_test['text'].apply(normazil)

date_new = []
for text in date_test['text'].tolist():
  text = text.split(' ')
  new_text = ""
  for word in text:
    if '@' not in word and '#' not in word:
      new_text += f'{word} '
  
  date_new.append(new_text)

date_test['new_text'] = date_new


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

after_tokenizer = tokenizer(list(date_test['new_text']), padding=True, truncation=True,return_tensors='pt')

datasets = TensorDataset(after_tokenizer['input_ids'], after_tokenizer["attention_mask"])
dataloader = DataLoader(datasets, 16)

from torch.nn.functional import softmax
outputs_new = []
for index, batch in enumerate(dataloader):
    input_ids, attention_mask = batch
    with torch.no_grad():  # Вимкнути обчислення градієнтів
        outputs = model(input_ids, attention_mask=attention_mask)
    probabilities = softmax(outputs.logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    outputs_new.extend(predicted_classes.numpy())



sample = pd.read_csv('sample_submission.csv')
sample["target"] = outputs_new

sample.to_csv('submission_1.csv', index=False)

from google.colab import files
files.download('submission_1.csv')

sample
