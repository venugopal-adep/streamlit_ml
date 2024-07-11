import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Data Science Explorer", layout="wide")

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
st.title("🚀 Statistical Learning vs Machine Learning")

st.markdown("""
Welcome to the Data Science Explorer! Embark on an exciting journey through the realms of 
Statistical Learning and Machine Learning. Discover the power of data analysis and prediction 
in this interactive adventure!
""")

# Functions
@st.cache_data
def generate_data():
    np.random.seed(np.random.randint(100))
    X = np.random.rand(100, 1) * 10
    y = 2 + 3 * X + np.random.randn(100, 1) * 2
    return X, y

def update_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_stats = LinearRegression().fit(X_train, y_train)
    model_ml = LinearRegression().fit(X_train, y_train)

    fig = px.scatter(x=X_train.squeeze(), y=y_train.squeeze(), trendline="ols",
                     labels={"x": "X", "y": "y"},
                     title="Machine Learning Prediction")
    fig.update_layout(template="plotly_dark")

    return fig, model_stats, model_ml, X_test, y_test

# Generate initial data
X, y = generate_data()
fig, model_stats, model_ml, X_test, y_test = update_models(X, y)

# Sidebar
st.sidebar.header("🎛️ Control Panel")
if st.sidebar.button("🔄 Generate New Data"):
    X, y = generate_data()
    fig, model_stats, model_ml, X_test, y_test = update_models(X, y)

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Learn", "🔬 Explore", "🏋️ Train", "🧪 Quiz"])

with tab1:
    st.header("Statistical Learning vs Machine Learning")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistical Learning")
        st.markdown("""
        Statistical Learning focuses on:
        - Understanding relationships between variables
        - Making inferences about populations
        - Hypothesis testing and model interpretation
        """)
    
    with col2:
        st.subheader("🤖 Machine Learning")
        st.markdown("""
        Machine Learning emphasizes:
        - Making accurate predictions
        - Handling complex, high-dimensional data
        - Automating decision-making processes
        """)

with tab2:
    st.header("🔍 Data Explorer")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical Insights")
        st.write(f"Intercept: {model_stats.intercept_[0]:.2f}")
        st.write(f"Coefficient: {model_stats.coef_[0][0]:.2f}")
    
    with col2:
        st.subheader("Machine Learning Performance")
        y_pred = model_ml.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        st.write(f"Mean Squared Error: {mse:.2f}")

with tab3:
    st.header("🏋️ Model Trainer")
    st.markdown("Adjust the parameters to see how they affect the model's performance!")
    
    col1, col2 = st.columns(2)
    with col1:
        noise_level = st.slider("Noise Level", 0.1, 5.0, 2.0, 0.1)
        slope = st.slider("True Slope", 0.5, 5.0, 3.0, 0.1)
    
    with col2:
        intercept = st.slider("True Intercept", -5.0, 5.0, 2.0, 0.1)
        n_samples = st.slider("Number of Samples", 50, 500, 100, 10)
    
    if st.button("Train Model"):
        X = np.random.rand(n_samples, 1) * 10
        y = intercept + slope * X + np.random.randn(n_samples, 1) * noise_level
        fig, model_stats, model_ml, X_test, y_test = update_models(X, y)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Estimated Parameters")
            st.write(f"Estimated Intercept: {model_stats.intercept_[0]:.2f}")
            st.write(f"Estimated Slope: {model_stats.coef_[0][0]:.2f}")
        
        with col2:
            st.subheader("True Parameters")
            st.write(f"True Intercept: {intercept:.2f}")
            st.write(f"True Slope: {slope:.2f}")



with tab4:
    st.header("🧪 Knowledge Quiz")
    
    questions = [
        {
            "question": "What is the main focus of Statistical Learning?",
            "options": ["Making accurate predictions", "Understanding relationships between variables", "Creating complex models"],
            "answer": 1,
            "explanation": "Statistical Learning primarily focuses on understanding relationships between variables. It's like being a detective for data! For example, a statistical learning approach might help us understand how different factors like exercise, diet, and genetics relate to a person's health. It's not just about predicting who might get sick, but understanding why and how these factors contribute to health outcomes."
        },
        {
            "question": "In linear regression, what does the coefficient represent?",
            "options": ["The y-intercept", "The slope of the line", "The prediction error"],
            "answer": 1,
            "explanation": "In linear regression, the coefficient represents the slope of the line. Think of it as the 'steepness' of your trend. For example, if you're looking at how ice cream sales relate to temperature, the coefficient might tell you that for every 1 degree increase in temperature, ice cream sales go up by $5. A steeper slope (higher coefficient) would mean a stronger relationship between temperature and ice cream sales."
        },
        {
            "question": "Which approach is more suitable for predicting house prices based on various features?",
            "options": ["Statistical Learning", "Machine Learning", "Both are equally suitable"],
            "answer": 1,
            "explanation": "Machine Learning is more suitable for predicting house prices based on various features. It's like having a super-smart real estate agent who can consider hundreds of factors at once! While statistical learning could help us understand how specific features (like number of bedrooms) affect price, machine learning can take into account complex interactions between many features (location, size, age, nearby schools, recent sales, etc.) to make more accurate predictions, especially when dealing with large amounts of data."
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

Congratulations on exploring the world of Data Science! Remember:

- 📊 Statistical learning helps us understand 'why' and 'how' variables are related.
- 🤖 Machine learning focuses on 'what' predictions we can make with the data.

Both approaches are powerful tools in the data scientist's toolkit. Keep exploring, keep learning, and may your models always converge! 🚀
""")
