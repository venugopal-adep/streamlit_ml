import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Train-Test Split Regressor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6e6;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
.highlight {
    background-color: #ffff00;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔀 Train-Test Split Regressor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore the impact of train-test split on regression models!")

# Helper functions
@st.cache_data
def generate_data(n_samples, n_features, noise):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

def plot_data(X, y, X_train, X_test, y_train, y_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', name='All Data', marker=dict(color='gray', size=8, opacity=0.5)))
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Training Data', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Testing Data', marker=dict(color='red', size=8)))
    fig.update_layout(
        title="Regression Data Visualization",
        xaxis_title="Feature",
        yaxis_title="Target",
        legend_title="Data Split"
    )
    return fig

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_predictions(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Training Data', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Testing Data', marker=dict(color='red', size=8)))
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_pred_train, mode='lines', name='Training Predictions', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_pred_test, mode='lines', name='Testing Predictions', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title="Regression Predictions",
        xaxis_title="Feature",
        yaxis_title="Target",
        legend_title="Data and Predictions"
    )
    return fig

# Sidebar
st.sidebar.title("Parameters")
n_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=500, value=200, step=50)
n_features = st.sidebar.slider("Number of Features", min_value=1, max_value=10, value=1, step=1)
noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=100.0, value=30.0, step=10.0)
test_size = st.sidebar.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

# Generate data
X, y = generate_data(n_samples, n_features, noise)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Data Visualization", "📈 Model Performance", "🧠 Quiz"])

with tab1:
    st.header("Understanding Train-Test Split in Regression")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Train-Test Split?</h3>
    <p>Train-test split is a technique used to evaluate the performance of a machine learning model. It involves dividing the dataset into two subsets: one for training the model and another for testing its performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts</h3>
    <ul>
        <li><strong>Training Set:</strong> The subset of data used to train the model</li>
        <li><strong>Testing Set:</strong> The subset of data used to evaluate the model's performance</li>
        <li><strong>Test Size:</strong> The proportion of data allocated to the testing set</li>
        <li><strong>Model Generalization:</strong> How well the model performs on unseen data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why is Train-Test Split Important?</h3>
    <ul>
        <li><span class="highlight">Assessing Model Performance:</span> Helps evaluate how well the model generalizes to new data</li>
        <li><span class="highlight">Preventing Overfitting:</span> Identifies if the model is memorizing the training data instead of learning general patterns</li>
        <li><span class="highlight">Model Selection:</span> Allows comparison of different models on the same test set</li>
        <li><span class="highlight">Hyperparameter Tuning:</span> Facilitates finding the best model parameters without using the test set</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Data Visualization")
    
    fig = plot_data(X, y, X_train, X_test, y_train, y_test)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    This plot shows the distribution of the data points:
    - Gray points represent all data
    - Blue points represent the training data
    - Red points represent the testing data
    """)
    
    fig_pred = plot_predictions(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.markdown("""
    This plot shows the model's predictions:
    - Solid points represent actual data
    - Dashed lines represent the model's predictions
    - Blue represents training data and predictions
    - Red represents testing data and predictions
    """)

with tab3:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Training Set Size", value=len(X_train))
        st.metric(label="Training MSE", value=f"{train_mse:.2f}")
        st.metric(label="Training R-squared", value=f"{train_r2:.2f}")
    with col2:
        st.metric(label="Testing Set Size", value=len(X_test))
        st.metric(label="Testing MSE", value=f"{test_mse:.2f}")
        st.metric(label="Testing R-squared", value=f"{test_r2:.2f}")
    
    st.markdown("""
    - **MSE (Mean Squared Error)**: Lower values indicate better model performance
    - **R-squared**: Ranges from 0 to 1, with 1 indicating perfect prediction
    
    Compare the metrics between training and testing sets to assess how well the model generalizes to new data.
    """)

with tab4:
    st.header("Test Your Knowledge")

    questions = [
        {
            "question": "What is the main purpose of train-test split?",
            "options": ["To increase the dataset size", "To evaluate model performance on unseen data", "To speed up model training"],
            "correct": 1,
            "explanation": "The main purpose of train-test split is to evaluate how well a model performs on unseen data, helping to assess its generalization capability."
        },
        {
            "question": "What does a larger test set typically provide?",
            "options": ["Faster model training", "More reliable performance estimates", "Better model accuracy"],
            "correct": 1,
            "explanation": "A larger test set typically provides more reliable performance estimates, as it gives a better representation of how the model might perform on new, unseen data."
        },
        {
            "question": "Which metric ranges from 0 to 1, with 1 indicating perfect prediction?",
            "options": ["Mean Squared Error (MSE)", "R-squared", "Test Size"],
            "correct": 1,
            "explanation": "R-squared ranges from 0 to 1, with 1 indicating perfect prediction. It represents the proportion of variance in the dependent variable that is predictable from the independent variable(s)."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Great job!")
            else:
                st.error("Not quite. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the impact of train-test split on regression models. Adjust the parameters and explore the different tabs to learn more!")
