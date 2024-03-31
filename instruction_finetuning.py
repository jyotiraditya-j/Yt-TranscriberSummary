from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

# Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set the padding token for the tokenizer
tokenizer.pad_token = tokenizer.eos_token  # You can use any token as the padding token, here we use eos_token

# Rest of your code goes here...


# Load the pre-trained model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2')

texts = ["I loved the movie. It was great!",
         "The food was terrible.",
         "The weather is okay."]
sentiments = ["positive", "negative", "neutral"]
instructions = ["Analyze the sentiment of the text and identify if it is positive.",
                "Analyze the sentiment of the text and identify if it is negative.",
                "Analyze the sentiment of the text and identify if it is neutral."]


# Tokenize the texts, sentiments, and instructions
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
encoded_instructions = tokenizer(instructions, padding=True, truncation=True, return_tensors='pt')

# Extract input IDs, attention masks, and instruction IDs
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']
instruction_ids = encoded_instructions['input_ids']

# To incorporate instructions during fine-tuning, we need to customize the model architecture. We can do this by concatenating the instruction IDs with the input IDs:

import torch

# Concatenate instruction IDs with input IDs and adjust attention mask
input_ids = torch.cat([instruction_ids, input_ids], dim=1)
attention_mask = torch.cat([torch.ones_like(instruction_ids), attention_mask], dim=1)

import torch.optim as optim

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Iterate over each input separately and its corresponding sentiment label
    for input_id, mask, sentiment in zip(input_ids, attention_mask, sentiments):
        outputs = model(input_id.unsqueeze(0), attention_mask=mask.unsqueeze(0))
        logits = outputs.logits
        # Convert sentiment label to numerical form
        sentiment_label = sentiments.index(sentiment)
        loss = criterion(logits, torch.tensor([sentiment_label]))  # Provide the sentiment label for each input
        loss.backward()
    optimizer.step()

