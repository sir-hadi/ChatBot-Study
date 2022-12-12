import nltk

# download pretrain tokenizer
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
# create stemmer object
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bags_of_words(tokenized_sentence, all_words):
    pass


sentence = "Hi there, my name is slim shady mcgrogri, whats yours? organize"
print(sentence)
print('tokenize :', tokenize(sentence))
# we dont need to make it an array it think
# maybe we can stem first then tokenize
# i wonder if i can make a stemmer from scratch, ok thats my next project
print('stem :', stem(sentence))


