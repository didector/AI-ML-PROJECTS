import streamlit as st
import pickle
import json
import random

# Load the trained model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input):
    # Preprocess user input and predict the intent
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    # Find the corresponding response
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            return response

    return "I'm sorry, I don't understand your question."

# Streamlit App UI
st.title("College Admission Chatbot")
st.write("Ask any question related to admissions, placements, or facilities.")

# User input field
user_input = st.text_input("You:", "")

# Generate response when input is provided
if user_input:
    response = chatbot_response(user_input)
    st.write(f"Chatbot: {response}")
