import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("C:\\Users\\luhar\\Desktop\\1234\\Restaurant_Reviews.tsv", sep='\t', quoting=3)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

# Preprocess reviews
data['Review'] = data['Review'].apply(preprocess_text)

# Vectorize the text
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(data['Review']).toarray()
y = data['Liked'].values

# Train a Naive Bayes model
classifier = GaussianNB()
classifier.fit(X, y)

# Streamlit UI
st.title('Restaurant Reviews Sentiment Classifier')
review_input = st.text_area('Enter a Restaurant Review')

if st.button('Predict Sentiment'):
    processed_review = preprocess_text(review_input)
    vectorized_review = cv.transform([processed_review]).toarray()
    prediction = classifier.predict(vectorized_review)
    if prediction[0] == 1:
        st.success("Positive Review")
    else:
        st.error("Negative Review")
