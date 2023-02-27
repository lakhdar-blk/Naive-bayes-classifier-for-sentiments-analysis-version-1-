import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download the stopwords corpus from NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Define the preprocessing function
def preprocess_message(text):

    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{3}[-.\s]??\d{3}[-.\s]??\d{4}', '', text)
    
    # Add negation tags to words that follow negations
    text = re.sub(r'\b(?:not|never|no|none|nobody|nothing|neither|nowhere)\b[\w\s]+[^\w\s]', lambda match: match.group().replace(' ', '_NEG '), text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text