import json
import pickle
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Lemmatization function
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in stopwords.words('english')]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Load intents JSON file
with open("C:/Users/ocn/OneDrive/Desktop/Allen_bot/allen.json", "r") as file:
    intents_data = json.load(file)

intents = intents_data["intents"]

# Prepare patterns and responses
patterns_responses = {}

for intent in intents:
    if "patterns" in intent and "responses" in intent:
        for pattern in intent["patterns"]:
            normalized_pattern = ' '.join(LemNormalize(pattern))
            patterns_responses[normalized_pattern] = intent["responses"]
    else:
        print(f"Missing patterns or responses in intent: {intent}")  # Debugging output

# Vectorization using TfidfVectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
tfidf_matrix = TfidfVec.fit_transform(patterns_responses.keys())

# Save the model and vectorizer using pickle
with open('Allenbot_model.pkl', 'wb') as model_file, open('Allenbot_tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(patterns_responses, model_file)  # Save response patterns
    pickle.dump(TfidfVec, tfidf_file)  # Save vectorizer

print("Model and vectorizer saved successfully!")
