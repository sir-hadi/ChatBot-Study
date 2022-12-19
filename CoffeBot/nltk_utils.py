import nltk
import numpy as np

# download pretrain tokenizer
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
# create stemmer object
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bags_of_words(tokenized_sentence, all_words):
    # ? the order of whne to stem? should we make all word into stem words too?
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag


# sentence = "Hi there, my name is slim shady mcgrogri, whats yours? organize"
# print(sentence)
# print('tokenize :', tokenize(sentence))
# # we dont need to make it an array it think
# # maybe we can stem first then tokenize
# # i wonder if i can make a stemmer from scratch, ok thats my next project
# print('stem :', stem(sentence))


