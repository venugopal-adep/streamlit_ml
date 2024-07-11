import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set page configuration
st.set_page_config(page_title="Bias-Variance Tradeoff Explorer", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        background-color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stTab {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🎢 Bias-Variance Tradeoff Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Bias-Variance Tradeoff Explorer! Dive into the world of model fitting and
understand the delicate balance between underfitting and overfitting. Interact with different
polynomial degrees and see how they affect model performance!
""")

# Functions
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
    fig = px.scatter(x=X.squeeze(), y=Y, labels={'x': 'X', 'y': 'Y'}, title="Model Fitting Demonstrations")
    fig.add_scatter(x=X_test.squeeze(), y=Y_test, mode='markers', name='Test Data')

    for degree, model in models.items():
        Y_plot = model.predict(X_plot)
        fig.add_scatter(x=X_plot.squeeze(), y=Y_plot, mode='lines', name=f'Degree {degree}')

    fig.update_layout(template="plotly_white")
    return fig

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Explore", "📊 Model Comparison", "📈 Metrics", "📚 Learn"])

with tab1:
    st.header("🔬 Data and Model Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        n_points = st.slider("Number of Data Points", 30, 150, 50)
    with col2:
        degrees = st.multiselect("Degrees of Polynomial", [1, 2, 3, 5, 10, 20], default=[1, 3, 10])
    
    if st.button("🔄 Generate New Data"):
        st.session_state['X'], st.session_state['Y'] = generate_data(n_points)
        st.success("New data generated!")
    
    if 'X' not in st.session_state or 'Y' not in st.session_state:
        st.session_state['X'], st.session_state['Y'] = generate_data(n_points)
    
    X_train, X_test, Y_train, Y_test = train_test_split(st.session_state['X'], st.session_state['Y'], test_size=0.2, random_state=42)
    models = fit_models(X_train, Y_train, degrees)
    
    fig = plot_models(X_train, Y_train, models, X_test, Y_test)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📊 Model Comparison")
    
    if 'X' in st.session_state and 'Y' in st.session_state:
        X_train, X_test, Y_train, Y_test = train_test_split(st.session_state['X'], st.session_state['Y'], test_size=0.2, random_state=42)
        models = fit_models(X_train, Y_train, degrees)
        
        for degree in degrees:
            st.subheader(f"Polynomial Degree: {degree}")
            X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
            Y_plot = models[degree].predict(X_plot)
            
            fig = px.scatter(x=X_train.squeeze(), y=Y_train, labels={'x': 'X', 'y': 'Y'})
            fig.add_scatter(x=X_test.squeeze(), y=Y_test, mode='markers', name='Test Data')
            fig.add_scatter(x=X_plot.squeeze(), y=Y_plot, mode='lines', name=f'Degree {degree} Model')
            fig.update_layout(template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please generate data in the Explore tab first.")

with tab3:
    st.header("📈 Model Metrics")
    
    if 'X' in st.session_state and 'Y' in st.session_state:
        X_train, X_test, Y_train, Y_test = train_test_split(st.session_state['X'], st.session_state['Y'], test_size=0.2, random_state=42)
        models = fit_models(X_train, Y_train, degrees)
        
        metrics_data = []
        for degree, model in models.items():
            train_score = r2_score(Y_train, model.predict(X_train))
            test_score = r2_score(Y_test, model.predict(X_test))
            metrics_data.append({"Degree": degree, "Train R²": train_score, "Test R²": test_score})
        
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df.style.format({"Train R²": "{:.2f}", "Test R²": "{:.2f}"}))
        
        fig = px.line(metrics_df, x="Degree", y=["Train R²", "Test R²"], title="R² Scores vs Polynomial Degree")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please generate data in the Explore tab first.")

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Understanding Bias-Variance Tradeoff")
    st.write("""
    The bias-variance tradeoff is a fundamental concept in machine learning:
    
    - **Bias**: The error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss relevant relations between features and target outputs (underfitting).
    
    - **Variance**: The error due to too much complexity in the learning algorithm. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).
    
    - **Tradeoff**: As we increase the complexity of our model, we'll see a reduction in error due to lower bias in the model. However, this complexity will also lead to an increase in error due to higher variance.
    """)
    
    st.subheader("Insights from the Demo")
    st.write("""
    - **Lower Degree Models**: May underfit the data (high bias, low variance). They're too simple to capture the underlying pattern.
    - **Higher Degree Models**: May overfit the data (low bias, high variance). They're so complex that they start to model the noise in the data.
    - **Optimal Degree**: Balances bias and variance to generalize well on unseen data. It captures the underlying pattern without fitting the noise.
    
    The goal is to find the sweet spot where the model complexity is just right - capturing the true underlying pattern without fitting to noise.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What typically happens to the training error as we increase model complexity?",
            "options": ["Increases", "Decreases", "Stays the same"],
            "answer": 1,
            "explanation": "As we increase model complexity (e.g., higher degree polynomials), the training error typically decreases. This is because a more complex model can fit the training data more closely. However, be cautious - a very low training error doesn't always mean a good model!"
        },
        {
            "question": "Which type of model is more likely to have high bias?",
            "options": ["A simple linear model", "A complex polynomial model", "They're equally likely to have high bias"],
            "answer": 0,
            "explanation": "A simple linear model is more likely to have high bias. Bias refers to the error introduced by approximating a real-world problem with a simplified model. Simple models often make strong assumptions about the data, which can lead to underfitting."
        },
        {
            "question": "What's the main risk of using a very high-degree polynomial model?",
            "options": ["Underfitting", "Overfitting", "It's always the best choice"],
            "answer": 1,
            "explanation": "The main risk of using a very high-degree polynomial model is overfitting. Such models can become so complex that they start fitting to the noise in the training data, rather than just the underlying pattern. This leads to poor generalization on new, unseen data."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(q["question"])
        user_answer = st.radio(f"Select your answer for question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['answer']:
                st.success("Correct! 🎉")
            else:
                st.error(f"Not quite. The correct answer is: {q['options'][q['answer']]}")
            
            st.markdown("**Explanation:**")
            st.write(q['explanation'])
            st.markdown("---")

st.markdown("""
## 🎓 Conclusion

Congratulations on exploring the Bias-Variance Tradeoff! Remember:

- 📊 The goal is to find the right balance between model simplicity and complexity.
- 🧮 A good model captures the underlying patterns without fitting to noise.
- 🚀 Keep exploring different models and always validate on unseen data!
""")
