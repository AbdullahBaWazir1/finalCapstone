import pandas as pd
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Load spaCy model
nlp = spacy.load('en_core_web_md')

# Load the dataset
database = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Extract the 'reviews.text' column
reviews_data = database['reviews.text'].dropna()

# Display the first few rows of the dataset
database.head()

# Display information about the dataset
database.info()

# Check for missing values in the dataset
database.isnull().sum()

# Filter the dataset to keep only relevant columns and drop rows with missing values
database = database[['reviews.date','reviews.text']].dropna()

# Display the first few rows of the filtered dataset
database.head()

# Tokenize the file.
"""
Tokenize the text using spaCy, lemmatize, and remove stopwords and punctuation.
"""
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# The File is huge, so Sample a subset to reduce the number of row for efficiency.
 
database = database.sample(1000, random_state=42)

# Transform review.text into numerical vectors using spaCy's word vectors.

def get_vector(text):
    doc = nlp(text)
    return doc.vector

# Apply the function to create a new column 'vector' in the dataset
database['vector'] = database['reviews.text'].apply(get_vector)
# Display the first few rows of the dataset with the 'vector' column added
database.head()

'''
Use TextBlob to analyze sentiment and determine whether each review expresses positive, negative, or neutral sentiment.
 
use the .sentiment and.polarity attribute 
to analyse the review and determine whether it 
expresses a positive, negative, or neutral sentiment.

'''
def analyze_sentiment_with_text(review):
    # Create a TextBlob object
    blob = TextBlob(review)
    # Determine sentiment label based on polarity score
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis with TextBlob to each review in the DataFrame
database['sentiment_text'] = database['reviews.text'].apply(analyze_sentiment_with_text)

# Display the DataFrame with sentiment analysis results from TextBlob
database.head()

# Example review for similarity comparison
review1 = database['reviews.text'][19947]
review1 

# Calculate similarity between tokens in review1
for token in review1:
    token = nlp(token)
    for token_ in review1:
        token_ = nlp(token_)
        print(token.similarity(token_))

# Example review for similarity comparison        
review2 = database['reviews.text'][11827]
review2

# Calculate similarity between tokens in review2
for token in review2:
    token = nlp(token)
    for token_ in review2:
        token_ = nlp(token_)
        print(token.similarity(token_))