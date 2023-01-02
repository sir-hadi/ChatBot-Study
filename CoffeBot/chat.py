import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, stem, bags_of_words

with open('intents.json', 'r') as file:
    intents = json.load(file)

# create model and put on model on GPU/CUDA if available
if torch.cuda.is_available():
    device = torch.device('cuda')

    print('there are %d GPU(s) available.' % torch.cuda.device_count())

    print('we will use the GPU: ', torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead")
    device = torch.device("cpu")

MODEL_FILENAME = "data.pth"
data = torch.load(MODEL_FILENAME)
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Asep"
print("Let's Chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bags_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X.to(device))
    logits, highest_logits_index = torch.max(output, dim=1)

    predicted_tag = tags[highest_logits_index.item()]

    probs = torch.softmax(output, dim=1)
    highest_prob = probs[0][highest_logits_index.item()]

    if highest_prob > 0.75:
        for intent in intents['intents']:
            if predicted_tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: idk 🤔")
