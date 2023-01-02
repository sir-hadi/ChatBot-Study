import json
import numpy as np
from nltk_utils import tokenize, stem, bags_of_words

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

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
all_words = [stem(word) for word in all_words if word not in ignore_words]
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
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample


# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])  # or len(all_words)
learning_rate = 0.001
nums_epochs = 1000

# Dataset object
'''
# ! if you have an error like below:

RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

# ! you must set the num of workers to zero in the data loader
# ! or just dont set the number of workers
'''
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True)

# create model and put on model on GPU/CUDA if available
if torch.cuda.is_available():
    device = torch.device('cuda')

    print('there are %d GPU(s) available.' % torch.cuda.device_count())

    print('we will use the GPU: ', torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead")
    device = torch.device("cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(nums_epochs):
    for (words, labels) in train_loader:
        # ? its possible that only just train_loader can be apply the to(device) function
        # ? not the words and labels, if that is possible we can have a cleaner code
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)

        # forward pass
        output = model(words)
        # ! error RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward, change the type then from
        # ! the change the variable of labels to long, we can use the argument in function to() dtype to fix this
        # ! we dont need to hange the words variable to float cause we already do that using numpy zeros and making it a float in the function bags_of_words
        loss = criterion(output, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{nums_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')
