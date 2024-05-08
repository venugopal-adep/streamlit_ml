import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

st.title('Machine Learning Model Deployment')

# Load iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# User input features
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 4.1)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.3)
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

input_data = user_input_features()

# Model prediction
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['target'])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(data.target_names))

st.subheader('Prediction')
st.write(data.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
