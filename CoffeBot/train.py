import json
import numpy as np
from nltk_utils import tokenize, stem, bags_of_words

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        '''
        # extend is awesome
        fruits = ['apple', 'banana', 'cherry']

        cars = ['Ford', 'BMW', 'Volvo']
        fruits.extend(cars)

        # fruits result will be as the following:
        #['apple', 'banana', 'cherry', 'Ford', 'BMW', 'Volvo']
        '''

        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [ stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bags_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_sample = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    # to get value by index on this object
    # we will use __getitem__
    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.n_sample

# Hyperparameters
batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, num_workers=6)