# https://blog.salesforceairesearch.com/learned-in-translation-contextualized-word-vectors/
# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
# https://towardsdatascience.com/understanding-nlp-word-embeddings-text-vectorization-1a23744f7223
# The process of converting words into numbers are called Vectorization.

#%%
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

with open('data/text.txt') as f:
    corpus = f.read()
    # corpus = word_tokenize(corpus)




# nltk.download('punkt')
# nltk.download('wordnet')

tokens = word_tokenize(corpus)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(token) for token in tokens]

print(lemmatized_words)



#%%

from sklearn.feature_extraction.text import CountVectorizer
# initialize
cv = CountVectorizer(stop_words='english') 
cv_matrix = cv.fit_transform(corpus) 
cv_matrix

#%%
# create document term matrix
data = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out())
data.to_csv('data/data_nlp.csv', index = False)


#%%
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(max_length,), dtype='int32')

# Define the embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)

# Define the LSTM layer
lstm_layer = LSTM(units=lstm_units, return_sequences=True)(embedding_layer)

# Define the attention layer
attention = TimeDistributed(Dense(1, activation='tanh'))(lstm_layer)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(lstm_units)(attention)
attention = Permute([2, 1])(attention)

# Combine the attention weights with the LSTM outputs
weighted = Multiply()([lstm_layer, attention])
weighted = Lambda(lambda x: K.sum(x, axis=1))(weighted)

# Define the output layer
output_layer = Dense(units=num_classes, activation='softmax')(weighted)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)


#%%








# %%
