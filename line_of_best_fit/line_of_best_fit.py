import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import time  # Used for varying the seed in data generation

# Function to generate random data
def generate_data():
    np.random.seed(int(time.time()))  # Use current time as seed for randomness
    total_bill = np.random.normal(20, 8, 100)
    tips = total_bill * 0.2 + np.random.normal(2, 1, 100)
    return pd.DataFrame({'total_bill': total_bill, 'tips': tips})

# Function to train model and predict
def train_and_predict(data):
    model = LinearRegression()
    model.fit(data[['total_bill']], data['tips'])
    predictions = model.predict(data[['total_bill']])
    return predictions, model

# Plot function with optional best fit line
def plot_data(data, predictions, show_best_fit):
    fig = px.scatter(data, x='total_bill', y='tips', opacity=0.65, title='Tip Prediction vs. Total Bill')
    if show_best_fit:
        # Best fit line
        fig.add_scatter(x=data['total_bill'], y=predictions, mode='lines', name='Best Fit Line', line=dict(color='blue'))
    return fig

st.write("## Finding the line of Best fit")
st.write("**Developed by : Venugopal Adep**")

# Initialize or get the state for the best fit line toggle
if 'show_best_fit' not in st.session_state:
    st.session_state.show_best_fit = True

# Sidebar for user interaction
st.sidebar.header('Data Controls')
if st.sidebar.button('Generate New Data'):
    data = generate_data()
    predictions, model = train_and_predict(data)
else:
    data = generate_data()
    predictions, model = train_and_predict(data)

# Button to toggle the best fit line
if st.sidebar.button('Toggle Best Fit Line'):
    st.session_state.show_best_fit = not st.session_state.show_best_fit

# Display the plot
fig = plot_data(data, predictions, st.session_state.show_best_fit)
st.plotly_chart(fig)

# Display regression equation and model accuracy
slope, intercept = model.coef_[0], model.intercept_
st.write("### Regression Equation")
st.latex(f"\\text{{Tips}} = ({slope:.2f}) \\times \\text{{Total Bill}} + ({intercept:.2f})")

# Explanation and model metrics
st.write("""
### Understanding the Model
- The linear regression model finds the line of best fit by minimizing the sum of squared residuals.
- Use the sidebar to toggle the best fit line on and off to compare the impact of the regression line.
""")
