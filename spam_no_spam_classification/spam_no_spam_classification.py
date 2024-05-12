import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import requests
import zipfile
import io

# Download the dataset from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)

# Extract the data from the ZIP archive
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    file_name = zip_ref.namelist()[0]  # Assumes the ZIP archive contains only one file
    with zip_ref.open(file_name) as file:
        data = pd.read_csv(file, sep="\t", names=["label", "message"])

# Preprocess the data
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_features)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Streamlit application
def main():
    st.write("## Email Spam Prediction")
    st.write("**Developed by : Venugopal Adep**")
    
    # User input for email message
    email_input = st.text_area("Enter an email message")
    
    # Sample email selection
    sample_emails = X_test.sample(n=5, random_state=42).tolist()
    sample_emails_with_labels = [(email, label) for email, label in zip(sample_emails, y_test[X_test.isin(sample_emails)])]
    selected_email = st.selectbox("Or select a sample email", [""] + [f"{email} ({'Spam' if label == 1 else 'Not Spam'})" for email, label in sample_emails_with_labels])
    
    if st.button("Predict"):
        if email_input:
            input_email = [email_input]
        elif selected_email:
            input_email = [selected_email.split(" (")[0]]  # Extract the email message from the selected option
        else:
            st.warning("Please enter an email message or select a sample email.")
            return
        
        # Preprocess the input email
        input_features = vectorizer.transform(input_email)
        
        # Predict using the trained model
        prediction = clf.predict(input_features)[0]
        
        # Display the prediction result
        if prediction == 1:
            st.error("This email is predicted to be spam.")
        else:
            st.success("This email is predicted to be not spam.")
    
    # Display model performance metrics
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    
    # Visualize the confusion matrix using Plotly
    cm_df = pd.DataFrame(cm, index=["Not Spam", "Spam"], columns=["Not Spam", "Spam"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig)

    # Display the first 10 rows of the dataset
    st.subheader("Dataset Preview")
    st.write(data.head(10))

if __name__ == '__main__':
    main()