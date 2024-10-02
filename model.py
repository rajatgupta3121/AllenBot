import nltk
import string
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Lemmatization functions
def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in nltk.corpus.stopwords.words('english')]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Load intents file
with open('allen.json', 'r') as file:
    data = json.load(file)

# Extract patterns and responses
patterns_responses = {}

for intent in data['intents']:
    patterns = intent['patterns']
    responses = intent['responses']
    
    # Store patterns and associated responses
    for pattern in patterns:
        normalized_pattern = ' '.join(LemNormalize(pattern))
        patterns_responses[normalized_pattern] = responses

# Create a list of patterns (keys)
preprocessed_patterns = list(patterns_responses.keys())

# Train TF-IDF Vectorizer
TfidfVec = TfidfVectorizer()
tfidf_matrix = TfidfVec.fit_transform(preprocessed_patterns)

# Save the model and vectorizer
with open('Allenbot_model.pkl', 'wb') as model_file, open('Allenbot_tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(patterns_responses, model_file)
    pickle.dump(TfidfVec, tfidf_file)

print("Model and vectorizer saved successfully.")
