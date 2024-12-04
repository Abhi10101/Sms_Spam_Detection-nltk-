import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum() and word not in stopwords.words('english')]
    # Apply stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")


input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input message
        transform_sms = transform_text(input_sms)

        # Debugging: Display the processed text
        st.write("Processed text:", transform_sms)

        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transform_sms])

        # Debugging: Display the vectorized input
        st.write("Vectorized input:", vector_input.toarray())

        # 3. Predict whether the message is spam or not
        result = model.predict(vector_input)[0]

        # Debugging: Get prediction probabilities
        result_prob = model.predict_proba(vector_input)[0]
        st.write("Prediction probabilities:", result_prob)

        # 4. Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")
