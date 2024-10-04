import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Spam Message Classification")

# Variables to track classifier
classifier = None
vectorizer = None

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your spam dataset (.tsv)", type=["tsv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, sep='\t')
    st.write("Dataset Preview:")
    st.write(df.head())

    # Data Preprocessing
    df.dropna(inplace=True)
    ham = df[df["label"] == "ham"]
    spam = df[df["label"] == "spam"]

    # Balance the dataset
    ham = ham.sample(spam.shape[0])
    data = pd.concat([ham, spam], ignore_index=True)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.3, random_state=0, shuffle=True)

    # Select classifier
    classifier_choice = st.selectbox("Choose a classifier", ["Random Forest", "SVC"])

    # Build the classifier pipeline
    if classifier_choice == "Random Forest":
        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=10))
        ])
    elif classifier_choice == "SVC":
        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('svm', SVC(C=100, gamma="auto"))
        ])

    # Train the model
    classifier.fit(x_train, y_train)

    # Model Evaluation
    y_pred = classifier.predict(x_test)
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
else:
    st.write("No dataset uploaded yet, using default Random Forest classifier.")

# Allow user to test with their own message, even before or after uploading the dataset
st.write("### Test with your own message")
user_message = st.text_area("Enter a message to classify (spam or ham):")

if user_message:
    if classifier is None:
        # If no dataset was uploaded, use a pre-trained simple Random Forest classifier
        vectorizer = TfidfVectorizer()
        default_data = {
            "message": [
                "You have won a lottery! Call now!",
                "Hey, are we still meeting for coffee tomorrow?",
                "Win an iPhone by clicking here!",
                "Looking forward to catching up soon!"
            ],
            "label": ["spam", "ham", "spam", "ham"]
        }
        df_default = pd.DataFrame(default_data)
        x_train_default = vectorizer.fit_transform(df_default["message"])
        y_train_default = df_default["label"]
        
        default_classifier = RandomForestClassifier(n_estimators=10)
        default_classifier.fit(x_train_default, y_train_default)

        # Predict user message with pre-trained default classifier
        user_vectorized = vectorizer.transform([user_message])
        user_pred = default_classifier.predict(user_vectorized)
    else:
        # Use the classifier trained with the uploaded dataset
        user_pred = classifier.predict([user_message])

    st.write(f"The message is classified as: **{user_pred[0]}**")
