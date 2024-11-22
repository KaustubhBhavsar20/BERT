# from flask import Flask, render_template, request
# import numpy as np
# import tensorflow as tf
# from transformers import BertTokenizer
# import pickle
# import re

# # Load the trained model and tokenizer
# MODEL_PATH = "my_bert_model.h5"
# TOKENIZER_PATH = "x_tokenizer.pkl"

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load the tokenizer for summaries (y_tokenizer)
# with open(TOKENIZER_PATH, 'rb') as f:
#     y_tokenizer = pickle.load(f)

# # Initialize BERT tokenizer for encoding input text
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Flask app initialization
# app = Flask(__name__)

# # Hyperparameters
# MAX_LEN_TEXT = 300
# MAX_LEN_SUMMARY = 100

# def preprocess_text(text):
#     """Preprocess input text."""
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#     return text

# def encode_text_with_bert(text, tokenizer, max_len):
#     """Tokenize and encode input text using the BERT tokenizer."""
#     encoded = tokenizer(
#         text,
#         padding="max_length",
#         truncation=True,
#         max_length=max_len,
#         return_tensors="tf"
#     )
#     return encoded["input_ids"]

# def summarize_text(input_text):
#     """Generate a summary for the given text."""
#     try:
#         # Preprocess and tokenize the input text
#         processed_text = preprocess_text(input_text)
#         encoded_text = encode_text_with_bert(processed_text, bert_tokenizer, MAX_LEN_TEXT)

#         # Predict summary using the model
#         prediction = model.predict([encoded_text, np.zeros((1, MAX_LEN_SUMMARY))])

#         # Convert predicted sequence to text using y_tokenizer
#         predicted_sequence = np.argmax(prediction, axis=-1)
#         summary = " ".join([
#             y_tokenizer.index_word.get(idx, "") for idx in predicted_sequence.flatten() if idx > 0
#         ])
        
#         # Debugging support
#         if not summary.strip():
#             print("Warning: Summary generation resulted in an empty string.")
#             print(f"Input text: {input_text}")
#             print(f"Processed text: {processed_text}")
#             print(f"Predicted sequence: {predicted_sequence}")

#         return summary.strip() or "Unable to generate a summary. Please try again."
#     except Exception as e:
#         print(f"Error during summary generation: {e}")
#         return "An error occurred during summarization."

# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html", summary="")

# @app.route("/summarize", methods=["POST"])
# def summarize():
#     input_text = request.form["text"]
#     summary = summarize_text(input_text)
#     return render_template("index.html", summary=summary)

# if __name__ == "__main__":
#     app.run(debug=True)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask import Flask, request, render_template
import json

app = Flask(__name__)

# Load the trained model
model = load_model('news_model.h5')  # Replace with the path to your saved model

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Function to process input text (tokenization + padding)
def process_input_text(input_text):
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([input_text])
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, padding='post')
    return padded_sequence

# Function to generate summary from model
def generate_summary(input_text):
    # Preprocess input text
    processed_input = process_input_text(input_text)
    
    # Assuming decoder_input_data should be prepared similarly
    # For simplicity, we'll assume it's the same as the input
    decoder_input_data = processed_input
    
    # Predict the summary
    prediction = model.predict([processed_input, decoder_input_data])
    
    # Convert the prediction back to text (using tokenizer)
    predicted_sequence = np.argmax(prediction[0], axis=-1)
    summary_words = tokenizer.sequences_to_texts([predicted_sequence])
    summary = summary_words[0]
    
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text_input']
        
        # Generate summary from the input text
        summary = generate_summary(input_text)
        
        return render_template('index.html', summary=summary)
    
    return render_template('index.html', summary=None)

if __name__ == '__main__':
    app.run(debug=True)
