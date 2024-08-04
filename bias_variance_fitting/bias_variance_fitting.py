import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set page config
st.set_page_config(page_title="Bias-Variance Tradeoff Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🎢 Bias-Variance Tradeoff Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the balance between model complexity and performance!")

# Helper functions
@st.cache_data
def generate_data(n_points):
    np.random.seed(42)
    X = np.linspace(0, 10, n_points)
    Y = np.sin(X) + np.random.normal(0, 0.5, n_points)  # Sine wave with noise
    X = X.reshape(-1, 1)
    return X, Y

def fit_models(X, Y, degrees):
    models = {}
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, Y)
        models[degree] = model
    return models

def plot_models(X, Y, models, X_test, Y_test):
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.squeeze(), y=Y, mode='markers', name='Training Data'))
    fig.add_trace(go.Scatter(x=X_test.squeeze(), y=Y_test, mode='markers', name='Test Data'))

    for degree, model in models.items():
        Y_plot = model.predict(X_plot)
        fig.add_trace(go.Scatter(x=X_plot.squeeze(), y=Y_plot, mode='lines', name=f'Degree {degree}'))

    fig.update_layout(title="Model Fitting Demonstrations", xaxis_title="X", yaxis_title="Y")
    return fig

# Sidebar
st.sidebar.header("Model Configuration")
n_points = st.sidebar.slider("Number of Data Points", 30, 150, 50)
degrees = st.sidebar.multiselect("Degrees of Polynomial", [1, 2, 3, 5, 10, 20], default=[1, 3, 10])

if st.sidebar.button("Generate New Data"):
    st.session_state['X'], st.session_state['Y'] = generate_data(n_points)

# Initialize session state
if 'X' not in st.session_state or 'Y' not in st.session_state:
    st.session_state['X'], st.session_state['Y'] = generate_data(n_points)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "🔬 Explore", "📊 Compare", "🧠 Quiz"])

with tab1:
    st.header("Understanding Bias-Variance Tradeoff")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is the Bias-Variance Tradeoff?</h3>
    <p>The bias-variance tradeoff is a fundamental concept in machine learning that deals with the balance between:</p>
    <ul>
        <li><strong>Bias:</strong> The error from overly simplistic assumptions in the learning algorithm.</li>
        <li><strong>Variance:</strong> The error from sensitivity to small fluctuations in the training set.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts</h3>
    <h4>1. Underfitting (High Bias)</h4>
    <p>When a model is too simple to capture the underlying pattern in the data.</p>
    <h4>2. Overfitting (High Variance)</h4>
    <p>When a model is too complex and starts fitting to noise in the training data.</p>
    <h4>3. Model Complexity</h4>
    <p>The degree of flexibility in the model, often related to the number of parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why is it Important?</h3>
    <ul>
        <li><span class="highlight">Model Selection:</span> Helps in choosing the right level of model complexity.</li>
        <li><span class="highlight">Generalization:</span> Ensures the model performs well on unseen data.</li>
        <li><span class="highlight">Performance Optimization:</span> Balances the tradeoff for optimal predictive performance.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("🔬 Data and Model Explorer")
    
    X_train, X_test, Y_train, Y_test = train_test_split(st.session_state['X'], st.session_state['Y'], test_size=0.2, random_state=42)
    models = fit_models(X_train, Y_train, degrees)
    
    fig = plot_models(X_train, Y_train, models, X_test, Y_test)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="background-color: #fffacd; padding: 10px; border-radius: 5px;">
    <p><strong>Interpretation:</strong> Observe how different polynomial degrees fit the data. 
    Lower degrees might underfit (high bias), while higher degrees might overfit (high variance).</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("📊 Model Comparison")
    
    metrics_data = []
    for degree, model in models.items():
        train_score = r2_score(Y_train, model.predict(X_train))
        test_score = r2_score(Y_test, model.predict(X_test))
        metrics_data.append({"Degree": degree, "Train R²": train_score, "Test R²": test_score})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig = px.line(metrics_df, x="Degree", y=["Train R²", "Test R²"], 
                  title="R² Scores vs Polynomial Degree",
                  labels={"value": "R² Score", "variable": "Dataset"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.table(metrics_df.style.format({"Train R²": "{:.2f}", "Test R²": "{:.2f}"}))
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 10px; border-radius: 5px;">
    <p><strong>Key Insight:</strong> Look for the degree where test R² is highest. 
    This is often the best balance between bias and variance.</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What happens to model complexity as we increase the polynomial degree?",
            "options": ["Decreases", "Increases", "Stays the same"],
            "correct": 1,
            "explanation": "As we increase the polynomial degree, the model becomes more complex, allowing it to fit more intricate patterns in the data."
        },
        {
            "question": "Which type of model is more likely to have high bias?",
            "options": ["A simple linear model", "A high-degree polynomial model", "They're equally likely to have high bias"],
            "correct": 0,
            "explanation": "A simple linear model is more likely to have high bias. It might be too simple to capture the underlying pattern in the data, leading to underfitting."
        },
        {
            "question": "What's a sign that a model might be overfitting?",
            "options": ["High training error, low test error", "Low training error, high test error", "Both training and test errors are high"],
            "correct": 1,
            "explanation": "Low training error but high test error is a classic sign of overfitting. The model has learned the training data too well, including its noise, and doesn't generalize to new data."
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
st.sidebar.info("This app demonstrates the bias-variance tradeoff. Adjust the settings, generate new data, and explore the different tabs to learn more!")
