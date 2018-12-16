#Utility class for generating batches of temporal data.
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import keras.utils as ku 
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from tensorflow import set_random_seed
from numpy.random import seed

# Comment By Sadhana Nandi : Initialize the random number generator to a constant value (7)
#This is important to ensure that the results we achieve from this model can be achieve again precisely.
#It ensures tha the stochastic process of training a neural network model can be reproduced.
set_random_seed(7)
seed(5)

import pandas as pandasVariable
import numpy as numpy
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
#Kapil's home directory - this should reflect your data set.
#curr_dir = '/Users/kapil_kumar/desktop/CS-410 Text Information Systems/Final Project/nyt-comments/'
curr_dir = '/Users/imac/Desktop/UIUC/cs410/Final Project/nyt-comments/'

all_news_headlines = []
for filename in os.listdir(curr_dir):
	if 'Articles' in filename:
	#if 'GameOfThrones' in filename:
		print(filename)
		# Update by Kapil :: Note - The dataset can be loaded directly. Since the output variable contains Strings,
		#It is easiest to load the data using pandas.
		article_df = pandasVariable.read_csv(curr_dir + filename)
		all_news_headlines.extend(list(article_df.headline.values))
		break
#article_df = pandasVariable.read_csv('result.txt', error_bad_lines=False)
def clean_text(txt):
	txt = "".join(v for v in txt if v not in string.punctuation).lower()
	txt = txt.encode("utf8").decode("ascii",'ignore')
	return txt
corpus = [clean_text(x) for x in all_news_headlines]

# Tokenizer is a text tokenization utility class provided by Keras.
#This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
#(Each integer being the index of a token in a dictionary), or into a vector where the coefficient for each token
#could be binary, based on word count, based on tf-idf etc.
tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1
	print(total_words)
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		print(token_list)
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			print(n_gram_sequence)
			input_sequences.append(n_gram_sequence)
	return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(inp_sequences[:6])
#Pads sequences to the same length.

#This function transforms a list of num_samples sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). 
#num_timesteps is either the maxlen argument if provided, or the length of the longest sequence otherwise.
#Sequences that are shorter than num_timesteps are padded with value at the end.
def create_padded_sequences(input_sequences):
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = numpy.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)
	return predictors, label, max_sequence_len
predictors, label, max_sequence_len = create_padded_sequences(inp_sequences)
print(create_padded_sequences(inp_sequences))
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 5, input_length=input_len))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    #Sadhana's comment: Before training a model, you need to configure the learning process
    #which is done via the compile method. The arguments are 'an optimizer', 'a loss function' and a list of metrics.
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model
model = create_model(max_sequence_len, total_words)
print(model.summary())
#Kapil's comment :: fit() Trains the model for a given number of epochs (iterations on a dataset).
model.fit(predictors, label, epochs=200, verbose=1)
def generate_fake_news_text(seed_text, next_words, model, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word,index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " "+output_word
	return seed_text.title()
usr_txt = raw_input("provide a word to get related news - ")
print (generate_fake_news_text(usr_txt,20, model, max_sequence_len))




