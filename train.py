import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('data.json', 'r') as f:
    intents = json.load(f)

# Hyperparameters
input_size = 100  # Adjust based on your tokenized input vector size
hidden_size = 128
output_size = len(intents['intents'])  # One class per intent
num_epochs = 1000
learning_rate = 0.001

# Preprocessing (Tokenizing and Lemmatizing)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

# Prepares a dataset of (X, y) where X is input (bag of words) and y is the intent class
def prepare_data():
    all_words = []
    tags = []
    data = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)

        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
            data.append((words, tag))
    
    all_words = [lemmatize(w) for w in all_words if w not in ['?', '.', '!']]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Convert sentences into bag-of-words format
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in data:
        bag = np.zeros(len(all_words), dtype=np.float32)
        for word in pattern_sentence:
            if lemmatize(word) in all_words:
                bag[all_words.index(lemmatize(word))] = 1.0
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, all_words, tags

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Prepare the dataset
X_train, y_train, all_words, tags = prepare_data()

# Create the model
model = NeuralNet(input_size=len(all_words), hidden_size=hidden_size, output_size=len(tags))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")

# Save the model and metadata
torch.save(model.state_dict(), "model.pth")

data = {
    "all_words": all_words,
    "tags": tags
}

with open("data.pth", "w") as f:
    json.dump(data, f)
