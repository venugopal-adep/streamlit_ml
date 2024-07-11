import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="OLS Regression Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("📈 Ordinary Least Squares (OLS) Regression Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of OLS regression in modeling linear relationships!")

# Helper functions
def generate_data(num_samples, slope, intercept, noise):
    X = np.random.rand(num_samples, 1)
    y = slope * X + intercept + noise * np.random.randn(num_samples, 1)
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Sidebar
st.sidebar.header("Model Parameters")
num_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, value=100, step=10)
slope = st.sidebar.slider("True Slope", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
intercept = st.sidebar.slider("True Intercept", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Generate data
X, y = generate_data(num_samples, slope, intercept, noise)

# Train model
model = train_model(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluation metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("OLS Regression in Action")
    
    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(
        title="Ordinary Least Squares (OLS) Regression",
        xaxis_title="X",
        yaxis_title="y",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estimated Slope", f"{model.coef_[0][0]:.4f}")
    with col2:
        st.metric("Estimated Intercept", f"{model.intercept_[0]:.4f}")
    with col3:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    with col4:
        st.metric("R-squared", f"{r2:.4f}")

with tab2:
    st.header("Solved Example: Effect of Noise")
    
    X_low_noise, y_low_noise = generate_data(100, 2, 1, 0.5)
    X_high_noise, y_high_noise = generate_data(100, 2, 1, 5)
    
    model_low_noise = train_model(X_low_noise, y_low_noise)
    model_high_noise = train_model(X_high_noise, y_high_noise)
    
    st.write("We'll compare OLS regression on two datasets with different noise levels.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Low Noise (0.5)")
        st.metric("Estimated Slope", f"{model_low_noise.coef_[0][0]:.4f}")
        st.metric("Estimated Intercept", f"{model_low_noise.intercept_[0]:.4f}")
        st.metric("R-squared", f"{r2_score(y_low_noise, model_low_noise.predict(X_low_noise)):.4f}")
    with col2:
        st.subheader("High Noise (5.0)")
        st.metric("Estimated Slope", f"{model_high_noise.coef_[0][0]:.4f}")
        st.metric("Estimated Intercept", f"{model_high_noise.intercept_[0]:.4f}")
        st.metric("R-squared", f"{r2_score(y_high_noise, model_high_noise.predict(X_high_noise)):.4f}")
    
    st.write("Notice how higher noise levels lead to less accurate estimates and lower R-squared values.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does OLS stand for in OLS Regression?",
            "options": ["Ordinary Least Squares", "Optimal Linear Solver", "Outlier Least Squares", "Orthogonal Linear Sampling"],
            "correct": 0,
            "explanation": "OLS stands for Ordinary Least Squares. It's a method for estimating the unknown parameters in a linear regression model by minimizing the sum of the squares of the differences between the observed and predicted values."
        },
        {
            "question": "What does R-squared measure in OLS Regression?",
            "options": ["The slope of the regression line", "The intercept of the regression line", "The proportion of variance in the dependent variable explained by the independent variable(s)", "The number of outliers in the dataset"],
            "correct": 2,
            "explanation": "R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model."
        },
        {
            "question": "What effect does increasing noise have on the R-squared value in OLS Regression?",
            "options": ["It increases R-squared", "It decreases R-squared", "It has no effect on R-squared", "It can either increase or decrease R-squared randomly"],
            "correct": 1,
            "explanation": "Increasing noise typically decreases the R-squared value. This is because noise introduces random variation that can't be explained by the model, reducing the proportion of variance that can be accounted for by the independent variable(s)."
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
    st.header("Learn More About OLS Regression")
    st.markdown("""
    Ordinary Least Squares (OLS) Regression is a statistical method for estimating the relationship between one or more independent variables and a dependent variable. It's one of the most basic and commonly used prediction techniques in statistics and machine learning.

    Key concepts in OLS Regression:
    1. **Linear Relationship**: OLS assumes a linear relationship between the independent and dependent variables.
    2. **Minimizing Squared Errors**: OLS finds the line that minimizes the sum of squared differences between observed and predicted values.
    3. **Assumptions**: OLS makes several assumptions, including linearity, independence, homoscedasticity, and normality of residuals.

    Mathematical Formula:
    The OLS model estimates parameters (slope and intercept) by solving:
    - Slope (β1) = Σ[(xi - mean(x)) * (yi - mean(y))] / Σ[(xi - mean(x))^2]
    - Intercept (β0) = mean(y) - β1 * mean(x)
    where (xi, yi) are data points, β1 is the slope, and β0 is the intercept.

    When to use OLS Regression:
    - When you want to understand the relationship between variables
    - When you want to make predictions based on linear relationships
    - When you have continuous variables and are interested in linear effects

    Remember, while OLS Regression is a powerful and widely used technique, it's important to check its assumptions and consider more advanced methods for complex relationships or when dealing with certain types of data.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates Ordinary Least Squares (OLS) Regression. Adjust the parameters and explore the different tabs to learn more!")
