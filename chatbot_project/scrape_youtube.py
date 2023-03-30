#%%
import yt_dlp as youtube_dl

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


url = 'https://www.youtube.com/watch?v=TpZcGhYp4rw'
download_audio(url)

# %%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load and tokenize the corpus of text
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()
    
inputs = tokenizer(corpus, return_tensors='pt')

# Train the model on the corpus of text
outputs = model(**inputs, labels=inputs['input_ids'])

# Compute the loss and backpropagate
loss = outputs.loss
loss.backward()
