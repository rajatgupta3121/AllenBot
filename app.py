from flask import Flask, request, jsonify, render_template
import pickle
import random
import nltk
import string
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os

# Download NLTK data (optional, can be removed if already downloaded)
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

# Load the model and vectorizer
with open('Allenbot_model.pkl', 'rb') as model_file, open('Allenbot_tfidf.pkl', 'rb') as tfidf_file:
    patterns_responses = pickle.load(model_file)
    TfidfVec = pickle.load(tfidf_file)

# Preprocess the patterns
preprocessed_patterns = list(patterns_responses.keys())
tfidf_matrix = TfidfVec.transform(preprocessed_patterns)

# Function to generate chatbot response
def get_response(user_input):
    preprocessed_input = ' '.join(LemNormalize(user_input))
    user_tfidf = TfidfVec.transform([preprocessed_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    max_sim_index = similarity_scores.argmax()

    if similarity_scores[0][max_sim_index] > 0.3:
        response = list(patterns_responses.values())[max_sim_index]
        return random.choice(response)
    else:
        return "I'm sorry, I don't understand. Can you please rephrase?"

# Text-to-speech (TTS) function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_path = "static/output.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error preparing audio: {e}")
        return None

# Flask web app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_message = request.form['message']
    bot_response = get_response(user_message.lower())

    # Generate audio for bot response
    audio_path = text_to_speech(bot_response)
    
    return jsonify(response=bot_response, audio_url=audio_path)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(host="0.0.0.0", port=8001, debug=True)
