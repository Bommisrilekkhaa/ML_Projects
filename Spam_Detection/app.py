import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os

# Get the absolute path to the current directory
current_directory = os.path.dirname(__file__)

# Specify the absolute path to "spam_clf.joblib"
clf_path = os.path.join(current_directory, "spam_clf.joblib")
vect_path = os.path.join(current_directory, "vectorizer.joblib")

# Print the file path for debugging
print(f"Attempting to load model from: {clf_path}")
print(f"Attempting to load vectorizer from: {vect_path}")

classifier = joblib.load(clf_path)

vect = joblib.load(vect_path)
print("Model loaded successfully.")

# Streamlit UI
st.title("Spam Classification App")

# User input for text
user_input = st.text_area("Enter a message:", "Type your message here...")

# Preprocess user input and make prediction
if st.button("Classify"):
    # Preprocess the user input
    user_input_processed = pd.Series(user_input)
    
    if 'vect' in locals():
        print("vect is ",(vect is not None))
    if 'vect' in locals() and vect is not None:
        x_user = vect.transform(user_input_processed)
        x_user_df = pd.DataFrame(x_user.toarray(), columns=vect.get_feature_names_out())

        # Make prediction
        prediction = classifier.predict(x_user_df)

        # Display the result
        if prediction[0] == 'ham':
            st.success("Prediction: Not Spam")
        else:
            st.error("Prediction: Spam")
    else:
        st.error("Error: Vectorizer is not defined.")
    
