import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

initialize_nltk()

def clean_text(text: str) -> str:
    """
    Preprocesses the input text for ML model.
    1. Lowercasing
    2. Removing punctuation
    3. Tokenization
    4. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
        
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens)
