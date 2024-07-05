import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from flask import Flask, request, jsonify, render_template
import numpy as np
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = load_model('next_word_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequence_length = 5  # Same as the length used during training

# Function to predict the next word
def predict_next_word(model, tokenizer, text_seq, seq_length):
    text_seq = text_seq.lower()
    text_seq = re.sub(r'[^a-z\s]', '', text_seq)  # Preprocess input text
    tokens = nltk.word_tokenize(text_seq)
    encoded = tokenizer.texts_to_sequences([tokens])
    encoded = pad_sequences(encoded, maxlen=seq_length-1, padding='pre')
    
    pred_prob = model.predict(encoded, verbose=0)
    pred_word = tokenizer.index_word[np.argmax(pred_prob)]
    
    return pred_word

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    next_word = predict_next_word(model, tokenizer, input_text, sequence_length)
    return jsonify({'next_word': next_word})

if __name__ == "__main__":
    app.run(debug=True)
