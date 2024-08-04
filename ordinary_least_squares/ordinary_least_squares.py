import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="OLS Regression Explorer", layout="wide", initial_sidebar_state="expanded")

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
    background-color: #e6e6fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Ordinary Least Squares (OLS) Regression Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of OLS regression in modeling linear relationships!")

# Helper functions
@st.cache_data
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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualization", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About OLS Regression")
    
    st.markdown("""
    <div class="highlight">
    <h3>What is OLS Regression?</h3>
    <p>Ordinary Least Squares (OLS) Regression is a statistical method used to find a line that best fits the relationship between an independent variable (X) and a dependent variable (y).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>How Does OLS Work?</h3>
    <ol>
        <li>It assumes a linear relationship between X and y.</li>
        <li>It finds the line that minimizes the sum of squared differences between observed and predicted y values.</li>
        <li>The line is described by two parameters: slope and intercept.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Concepts in OLS:</h3>
    <ul>
        <li><b>Slope:</b> The change in y for a one-unit change in X.</li>
        <li><b>Intercept:</b> The predicted value of y when X is zero.</li>
        <li><b>R-squared:</b> A measure of how well the model fits the data (ranges from 0 to 1).</li>
        <li><b>Mean Squared Error (MSE):</b> The average squared difference between predicted and actual y values.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 OLS Regression in Action")
    
    # Create scatter plot
    fig = px.scatter(x=X.flatten(), y=y.flatten(), labels={'x': 'X', 'y': 'y'}, title="OLS Regression")
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred.flatten(), mode='lines', name='Regression Line'))
    fig.update_layout(showlegend=True)
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
    
    st.markdown("""
    <div class="highlight">
    <p>The scatter plot shows the actual data points and the regression line. The metrics below show how well the model fits the data.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("🧮 Example: Effect of Noise")
    
    X_low_noise, y_low_noise = generate_data(100, 2, 1, 0.5)
    X_high_noise, y_high_noise = generate_data(100, 2, 1, 5)
    
    model_low_noise = train_model(X_low_noise, y_low_noise)
    model_high_noise = train_model(X_high_noise, y_high_noise)
    
    st.write("Let's compare OLS regression on two datasets with different noise levels.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_low = px.scatter(x=X_low_noise.flatten(), y=y_low_noise.flatten(), labels={'x': 'X', 'y': 'y'}, title="Low Noise (0.5)")
        fig_low.add_trace(go.Scatter(x=X_low_noise.flatten(), y=model_low_noise.predict(X_low_noise).flatten(), mode='lines', name='Regression Line'))
        st.plotly_chart(fig_low, use_container_width=True)
        st.metric("R-squared", f"{r2_score(y_low_noise, model_low_noise.predict(X_low_noise)):.4f}")
    with col2:
        fig_high = px.scatter(x=X_high_noise.flatten(), y=y_high_noise.flatten(), labels={'x': 'X', 'y': 'y'}, title="High Noise (5.0)")
        fig_high.add_trace(go.Scatter(x=X_high_noise.flatten(), y=model_high_noise.predict(X_high_noise).flatten(), mode='lines', name='Regression Line'))
        st.plotly_chart(fig_high, use_container_width=True)
        st.metric("R-squared", f"{r2_score(y_high_noise, model_high_noise.predict(X_high_noise)):.4f}")
    
    st.markdown("""
    <div class="highlight">
    <p>Notice how higher noise levels lead to a lower R-squared value and a less accurate fit. The regression line in the high noise plot doesn't capture the trend as well as in the low noise plot.</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does the slope in OLS regression represent?",
            "options": ["The starting point of the line", "How much y changes when X changes by 1", "The error in the model", "The number of data points"],
            "correct": 1,
            "explanation": "The slope represents how much the dependent variable (y) changes when the independent variable (X) increases by one unit."
        },
        {
            "question": "What does a higher R-squared value mean?",
            "options": ["The model is more complex", "The line is steeper", "The model fits the data better", "There's more noise in the data"],
            "correct": 2,
            "explanation": "A higher R-squared value (closer to 1) indicates that the model explains more of the variability in the data, meaning it fits the data better."
        },
        {
            "question": "How does increasing noise affect the OLS regression model?",
            "options": ["It makes the model fit better", "It has no effect", "It makes the model fit worse", "It always increases the slope"],
            "correct": 2,
            "explanation": "Increasing noise typically makes the model fit worse because it introduces random variation that can't be explained by the linear relationship."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Not quite right. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates Ordinary Least Squares (OLS) Regression. Adjust the parameters and explore the different tabs to learn more!")
