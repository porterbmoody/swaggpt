import pandas as pd
import sklearn


class SwagGPT():
    
    def __init__(self) -> None:
        self.user_responses = []
        self.bot_responses = []

        self.conversation = pd.DataFrame({
            'user_responses':self.user_responses,
            'bot_responses' :self.bot_responses
        })

    def get_response(self, user_response):
        if user_response == 'hello':
            bot_response = "hello how are you?"
        elif user_response == "pizza":
            bot_response = "i like pizza"
        elif user_response == "update":
            bot_response = "updating data..."
            self.update_data()
        elif user_response in self.conversation['user_responses'].values:
            print("user_response found:" + user_response)
            row = self.conversation.loc[self.conversation['user_responses'] == user_response]
            bot_response = "is this an appropriate response?" + row['bot_responses'].values[0]
        else:
            bot_response = "what would be a typical response to: " + user_response
        return bot_response
    
    def update_data(self, user_response, bot_response):
        if user_response != "" and bot_response != "":
            self.user_responses.append(user_response)        
            self.bot_responses.append(bot_response)
            print("RESPONSE DATA\n")
            self.conversation = pd.DataFrame({
                'user_responses':self.user_responses,
                'bot_responses' :self.bot_responses
            })

            print(self.conversation)
            print()

    def current_data(self):
        print("current responses:")
        print(self.user_responses)
        print(self.bot_responses)




#%%


import gensim
from gensim.models import Word2Vec

# Example sentence
sentence = "The quick brown fox jumped over the lazy dog."

# Tokenize the sentence
tokens = gensim.utils.simple_preprocess(sentence)

# Train a Word2Vec model on the tokens
model = Word2Vec([tokens], min_count=1, vector_size=100)

# Get the embedding for each token in the sentence
embeddings = [model.wv[word] for word in tokens]

# Print the embeddings for each token
for word, embedding in zip(tokens, embeddings):
    print(word, embedding)

# %%
