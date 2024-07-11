import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Set page config
st.set_page_config(page_title="Regularization Explorer", layout="wide", initial_sidebar_state="expanded")

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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔧 Regularization Techniques Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of Lasso and Ridge regularization in linear regression!")

# Helper functions
def generate_data(n_samples=100, n_features=10, noise=0.1, seed=42):
    np.random.seed(seed)
    X, y, coef = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, coef=True, random_state=seed)
    return X, y, coef

def fit_model(X_train, y_train, alpha, model_type='lasso'):
    if model_type == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def plot_coefficients(coefs, true_coefs, model_type):
    df = pd.DataFrame({
        'True Coefficients': true_coefs,
        'Estimated Coefficients': coefs
    })
    fig = px.bar(df, barmode='group', title=f'{model_type.capitalize()} Regression Coefficients Comparison')
    return fig

# Sidebar
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Select Regularization Type", ['lasso', 'ridge'])
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 1.0, 0.1, 0.01)

# Generate data once
if 'X' not in st.session_state:
    X, y, true_coefs = generate_data()
    st.session_state['X'], st.session_state['y'], st.session_state['true_coefs'] = X, y, true_coefs

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Regularization in Action")
    
    X, y, true_coefs = st.session_state['X'], st.session_state['y'], st.session_state['true_coefs']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = fit_model(X_train, y_train, alpha, model_type)
    
    # Display the specific regularization equation
    if model_type == 'lasso':
        st.latex(r"Lasso: \min_{\beta} \left\{ \frac{1}{2n} \left\| y - X\beta \right\|^2 + \alpha \left\| \beta \right\|_1 \right\}")
    else:
        st.latex(r"Ridge: \min_{\beta} \left\{ \frac{1}{2n} \left\| y - X\beta \right\|^2 + \alpha \left\| \beta \right\|^2_2 \right\}")
    
    # Plot coefficients
    predicted_coefs = model.coef_
    fig = plot_coefficients(predicted_coefs, true_coefs, model_type)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test MSE", f"{test_mse:.4f}")
    with col2:
        st.metric("Train MSE", f"{train_mse:.4f}")
    with col3:
        st.metric("Number of non-zero coefficients", np.sum(np.abs(predicted_coefs) > 1e-5))

with tab2:
    st.header("Solved Example: Lasso vs Ridge")
    
    X_example, y_example, true_coefs_example = generate_data(n_samples=50, n_features=5, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X_example, y_example, test_size=0.2, random_state=42)
    
    lasso_model = fit_model(X_train, y_train, alpha=0.1, model_type='lasso')
    ridge_model = fit_model(X_train, y_train, alpha=0.1, model_type='ridge')
    
    st.write("We'll compare Lasso and Ridge regression on a small dataset (50 samples, 5 features).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lasso Regression")
        st.metric("Test MSE", f"{mean_squared_error(y_test, lasso_model.predict(X_test)):.4f}")
        st.metric("Non-zero coefficients", np.sum(np.abs(lasso_model.coef_) > 1e-5))
    with col2:
        st.subheader("Ridge Regression")
        st.metric("Test MSE", f"{mean_squared_error(y_test, ridge_model.predict(X_test)):.4f}")
        st.metric("Non-zero coefficients", np.sum(np.abs(ridge_model.coef_) > 1e-5))
    
    st.write("Notice how Lasso tends to produce sparse models by setting some coefficients exactly to zero, while Ridge shrinks all coefficients but rarely sets them exactly to zero.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main difference between Lasso and Ridge regularization?",
            "options": ["Lasso uses L1 penalty, Ridge uses L2 penalty", "Lasso is for classification, Ridge is for regression", "Lasso is faster, Ridge is more accurate", "Lasso uses gradient descent, Ridge uses normal equations"],
            "correct": 0,
            "explanation": "The main difference is in the penalty term: Lasso uses L1 regularization (absolute value of coefficients), which can lead to sparse models, while Ridge uses L2 regularization (squared value of coefficients), which tends to shrink coefficients towards zero but not exactly to zero."
        },
        {
            "question": "What does the alpha parameter control in regularization?",
            "options": ["The learning rate", "The number of iterations", "The strength of regularization", "The number of features"],
            "correct": 2,
            "explanation": "Alpha controls the strength of regularization. A higher alpha value increases the impact of the regularization term, leading to stronger shrinkage of coefficients."
        },
        {
            "question": "Which regularization technique is more likely to produce a sparse model?",
            "options": ["Lasso", "Ridge", "Both equally", "Neither"],
            "correct": 0,
            "explanation": "Lasso regularization is more likely to produce sparse models. It has the effect of forcing some of the coefficient estimates to be exactly zero when the tuning parameter alpha is sufficiently large, effectively performing feature selection."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
            st.write(f"Explanation: {q['explanation']}")
        st.write("---")

with tab4:
    st.header("Learn More About Regularization")
    st.markdown("""
    Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. It discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.

    Key benefits of Regularization:
    1. **Prevents Overfitting**: Helps the model generalize better to unseen data.
    2. **Feature Selection**: Lasso can perform automatic feature selection.
    3. **Multicollinearity**: Ridge regression can handle multicollinearity in the data.

    Types of Regularization:
    1. **Lasso (L1)**: Adds absolute value of magnitude of coefficients as penalty term to the loss function.
    2. **Ridge (L2)**: Adds squared magnitude of coefficients as penalty term to the loss function.
    3. **Elastic Net**: Combines both L1 and L2 penalties.

    When to use which:
    - Use Lasso when you believe many features are irrelevant.
    - Use Ridge when you have many small/medium sized effects.
    - Use Elastic Net when you have many correlated features.

    Remember, while regularization is a powerful technique for preventing overfitting, it's not a silver bullet. It's important to understand the nature of your data and the problem you're trying to solve when choosing and tuning regularization techniques.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates Lasso and Ridge regularization techniques. Adjust the settings and explore the different tabs to learn more!")
