from flask import Flask, render_template, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure that necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('wordnet')

app = Flask(__name__)

# Load intents from JSON
with open('data.json') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Tokenization and lemmatization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

# Find intent based on input
def find_intent(user_input):
    tokens = tokenize(user_input)
    lemmatized_tokens = [lemmatize(token) for token in tokens]
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = tokenize(pattern)
            if all(lemmatize(token) in lemmatized_tokens for token in pattern_tokens):
                return intent
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    intent = find_intent(user_input)
    
    if intent:
        response = random.choice(intent['responses'])
    else:
        response = "I don't understand. Can you please rephrase?"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
