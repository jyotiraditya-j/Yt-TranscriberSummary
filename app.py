import os
import warnings

import openai
import torch
import transformers

warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "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"

from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

# Load the pre-trained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the pre-trained model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


texts= ["I loved the movie! It was great!",
        "The food was disgusting and terrible.",
        "The weather was okay."]

sentiments=["positive","negative","neutral"]

# Tokenize the text samples
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Extract the input IDs and attention masks
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']

# Convert the sentiment labels to numerical form
sentiment_labels = [sentiments.index(sentiment) for sentiment in sentiments]

#adding a simple linear layer to perform senti analysis

import torch.nn as nn

num_classes=len(set(sentiment_labels))

classification_head=nn.Linear(model.config.hidden_size, num_classes)

# # Replace the pre-trained model's classification head with our custom head
model.classifier = classification_head

import torch.optim as optim

optimizer = optim.AdamW(model.parameters(),lr=2e-5)
criterion=nn.CrossEntropyLoss()
#fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # Obtain the logits from the model output
    loss = criterion(logits, torch.tensor(sentiment_labels))  # Calculate the loss
    loss.backward()
    optimizer.step()
    optimizer.step()