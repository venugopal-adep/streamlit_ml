import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Set page config
st.set_page_config(page_title="Regularization Techniques Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #ffd700;
    padding: 5px;
    border-radius: 3px;
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
        'Feature': [f'X{i}' for i in range(len(coefs))],
        'True Coefficients': true_coefs,
        'Estimated Coefficients': coefs
    })
    df_melted = pd.melt(df, id_vars=['Feature'], var_name='Coefficient Type', value_name='Value')
    fig = px.bar(df_melted, x='Feature', y='Value', color='Coefficient Type', barmode='group',
                 title=f'{model_type.capitalize()} Regression Coefficients Comparison')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Coefficient Value')
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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualize", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("Understanding Regularization")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Regularization?</h3>
    <p>Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function:</p>
    <ul>
        <li>It discourages learning a more complex or flexible model.</li>
        <li>It helps the model generalize better to unseen data.</li>
        <li>It's like adding a "cost" for complexity to your model.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts in Regularization</h3>
    <h4>1. Lasso Regularization (L1)</h4>
    <p>Adds the absolute value of the magnitude of coefficients as a penalty term to the loss function.</p>
    <h4>2. Ridge Regularization (L2)</h4>
    <p>Adds the squared magnitude of coefficients as a penalty term to the loss function.</p>
    <h4>3. Alpha Parameter</h4>
    <p>Controls the strength of regularization. Higher alpha means stronger regularization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Regularization?</h3>
    <ul>
        <li><span class="highlight">Prevent Overfitting:</span> It helps your model perform well on new, unseen data.</li>
        <li><span class="highlight">Feature Selection:</span> Lasso can automatically select important features.</li>
        <li><span class="highlight">Handle Multicollinearity:</span> Ridge can handle correlated features effectively.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
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
        st.metric("Non-zero coefficients", np.sum(np.abs(predicted_coefs) > 1e-5))
    
    st.markdown("""
    <div style="background-color: #fffacd; padding: 10px; border-radius: 5px;">
    <p><strong>Interpretation:</strong> The bar chart compares true coefficients with estimated coefficients. 
    Lasso tends to push some coefficients to exactly zero, while Ridge shrinks all coefficients but rarely to zero.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Solved Example: Lasso vs Ridge")
    
    X_example, y_example, true_coefs_example = generate_data(n_samples=50, n_features=5, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X_example, y_example, test_size=0.2, random_state=42)
    
    lasso_model = fit_model(X_train, y_train, alpha=0.1, model_type='lasso')
    ridge_model = fit_model(X_train, y_train, alpha=0.1, model_type='ridge')
    
    st.write("Let's compare Lasso and Ridge regression on a small dataset (50 samples, 5 features).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lasso Regression")
        lasso_mse = mean_squared_error(y_test, lasso_model.predict(X_test))
        st.metric("Test MSE", f"{lasso_mse:.4f}")
        st.metric("Non-zero coefficients", np.sum(np.abs(lasso_model.coef_) > 1e-5))
        
        fig_lasso = plot_coefficients(lasso_model.coef_, true_coefs_example, 'Lasso')
        st.plotly_chart(fig_lasso, use_container_width=True)
    
    with col2:
        st.subheader("Ridge Regression")
        ridge_mse = mean_squared_error(y_test, ridge_model.predict(X_test))
        st.metric("Test MSE", f"{ridge_mse:.4f}")
        st.metric("Non-zero coefficients", np.sum(np.abs(ridge_model.coef_) > 1e-5))
        
        fig_ridge = plot_coefficients(ridge_model.coef_, true_coefs_example, 'Ridge')
        st.plotly_chart(fig_ridge, use_container_width=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Observations:</h3>
    <ul>
        <li>Lasso tends to produce sparse models by setting some coefficients exactly to zero.</li>
        <li>Ridge shrinks all coefficients but rarely sets them exactly to zero.</li>
        <li>The choice between Lasso and Ridge often depends on your specific dataset and problem.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does regularization help prevent in machine learning models?",
            "options": ["Overfitting", "Underfitting", "Data collection", "Model deployment"],
            "correct": 0,
            "explanation": "Regularization helps prevent overfitting by adding a penalty for model complexity, allowing the model to generalize better to unseen data."
        },
        {
            "question": "Which regularization technique is more likely to produce a sparse model?",
            "options": ["Lasso", "Ridge", "Both equally", "Neither"],
            "correct": 0,
            "explanation": "Lasso regularization is more likely to produce sparse models by forcing some coefficients to be exactly zero, effectively performing feature selection."
        },
        {
            "question": "What does a higher alpha value in regularization typically mean?",
            "options": ["Stronger regularization", "Weaker regularization", "Faster training", "More features"],
            "correct": 0,
            "explanation": "A higher alpha value typically means stronger regularization, increasing the impact of the penalty term and leading to more coefficient shrinkage."
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
st.sidebar.info("This app demonstrates Lasso and Ridge regularization techniques. Adjust the settings and explore the different tabs to learn more!")
