import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the saved model
classifier = joblib.load("./spam_clf.joblib")
vect = joblib.load("./vectorizer.joblib")
# Streamlit UI
st.title("Spam Classification App")

# User input for text
user_input = st.text_area("Enter a message:", "Type your message here...")

# Preprocess user input and make prediction
if st.button("Classify"):
    # Preprocess the user input
    user_input_processed = pd.Series(user_input)
    x_user = vect.transform(user_input_processed)
    x_user_df = pd.DataFrame(x_user.toarray(), columns=vect.get_feature_names_out())

    # Make prediction
    prediction = classifier.predict(x_user_df)

    # Display the result
    if prediction[0] == 'ham':
        st.success("Prediction: Not Spam")
    else:
        st.error("Prediction: Spam")
