import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords and punctuation
stopwords_set = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for preprocessing text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stopwords_set and token not in punctuation]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df['selected_text'] = df['selected_text'].astype(str)
    df['preprocessed_text'] = df['selected_text'].apply(preprocess_text)
    return df
