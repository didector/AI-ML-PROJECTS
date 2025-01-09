import nltk
import random
import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Load intents data
with open('dataset/intents1.json') as file:
    intents = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

# Print training data
print("Training Sentences:")
print(training_sentences)
print("Training Labels:")
print(training_labels)

# Vectorization and model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)
y = np.array(training_labels)

# Create model pipeline
models = {
    'logistic_regression': LogisticRegression(),
    'naive_bayes': MultinomialNB(),
    'svc': SVC(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier()
}

# Define hyperparameters for grid search
param_grid = {
    'logistic_regression': {'C': [0.1, 1, 10], 'max_iter': [100, 200], 'penalty': ['l2'], 'solver': ['liblinear']},
    'naive_bayes': {'alpha': [0.1, 0.5, 1]},
    'svc': {'C': [0.1, 1, 10], 'kernel': ['linear'], 'max_iter': [100, 200]},
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
    'random_forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
}

# Train each model with grid search
best_model = None
best_accuracy = 0
best_model_name = ''
best_params = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(model, param_grid[model_name], cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Accuracy of {model_name}: {grid_search.best_score_:.4f}")

    # Update best model if needed
    if grid_search.best_score_ > best_accuracy:
        best_accuracy = grid_search.best_score_
        best_model_name = model_name
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

# Print best model and its accuracy
print(f"\nBest Model: {best_model_name}")
print(f"Best Model Accuracy: {best_accuracy:.4f}")
print(f"Best Model Parameters: {best_params}")

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the best model and vectorizer
with open('models/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('models/chatbot_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("\nModel and Vectorizer saved successfully!")
