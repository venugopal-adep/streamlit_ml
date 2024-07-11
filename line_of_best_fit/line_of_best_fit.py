import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import time

# Set page configuration
st.set_page_config(page_title="Tip Predictor: Best Fit Line Explorer", layout="wide")

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
st.title("💰 Tip Predictor: Best Fit Line Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Tip Predictor! Explore the relationship between total bill amounts and tips
using linear regression. Interact with the data, toggle the best fit line, and understand
how linear regression works in predicting tips!
""")

# Functions
@st.cache_data
def generate_data():
    np.random.seed(int(time.time()))  # Use current time as seed for randomness
    total_bill = np.random.normal(20, 8, 100)
    tips = total_bill * 0.2 + np.random.normal(2, 1, 100)
    return pd.DataFrame({'total_bill': total_bill, 'tips': tips})

def train_and_predict(data):
    model = LinearRegression()
    model.fit(data[['total_bill']], data['tips'])
    predictions = model.predict(data[['total_bill']])
    return predictions, model

def plot_data(data, predictions, show_best_fit):
    fig = px.scatter(data, x='total_bill', y='tips', opacity=0.65, title='Tip Prediction vs. Total Bill')
    if show_best_fit:
        fig.add_scatter(x=data['total_bill'], y=predictions, mode='lines', name='Best Fit Line', line=dict(color='blue'))
    fig.update_layout(template="plotly_white")
    return fig

# Initialize session state
if 'show_best_fit' not in st.session_state:
    st.session_state.show_best_fit = True
if 'data' not in st.session_state:
    st.session_state.data = generate_data()
    st.session_state.predictions, st.session_state.model = train_and_predict(st.session_state.data)

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Explore", "📊 Model", "📈 Metrics", "📚 Learn"])

with tab1:
    st.header("🔬 Data Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Generate New Data"):
            st.session_state.data = generate_data()
            st.session_state.predictions, st.session_state.model = train_and_predict(st.session_state.data)
            st.success("New data generated and model retrained!")
    
    with col2:
        st.session_state.show_best_fit = st.checkbox("Show Best Fit Line", value=st.session_state.show_best_fit)
    
    fig = plot_data(st.session_state.data, st.session_state.predictions, st.session_state.show_best_fit)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Raw Data"):
        st.write(st.session_state.data)

with tab2:
    st.header("📊 Model Details")
    
    slope, intercept = st.session_state.model.coef_[0], st.session_state.model.intercept_
    
    st.subheader("Regression Equation")
    st.latex(f"\\text{{Tips}} = ({slope:.2f}) \\times \\text{{Total Bill}} + ({intercept:.2f})")
    
    st.subheader("Interpretation")
    st.write(f"""
    - For every $1 increase in the total bill, the tip is expected to increase by ${slope:.2f}.
    - When the total bill is $0, the expected tip is ${intercept:.2f} (this is the y-intercept).
    """)
    
    st.subheader("Prediction Example")
    example_bill = st.slider("Select a total bill amount", 0.0, 100.0, 50.0, 0.1)
    predicted_tip = slope * example_bill + intercept
    st.write(f"For a total bill of ${example_bill:.2f}, the predicted tip is ${predicted_tip:.2f}")

with tab3:
    st.header("📈 Model Metrics")
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_true = st.session_state.data['tips']
    y_pred = st.session_state.predictions
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    
    st.subheader("Residual Plot")
    residuals = y_true - y_pred
    fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Tips', 'y': 'Residuals'})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals.update_layout(title="Residual Plot", template="plotly_white")
    st.plotly_chart(fig_residuals, use_container_width=True)

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Understanding Linear Regression")
    st.write("""
    Linear regression is a statistical method used to model the relationship between two variables by fitting a linear equation to observed data. Here's what you need to know:
    
    1. **Best Fit Line**: The line that minimizes the sum of squared differences between observed and predicted values.
    2. **Slope**: Represents the change in the dependent variable (tips) for a one-unit change in the independent variable (total bill).
    3. **Intercept**: The expected value of tips when the total bill is zero (often not meaningful in real-world contexts).
    4. **R² Score**: Measures the proportion of variance in the dependent variable that is predictable from the independent variable.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What does a positive slope in our model indicate?",
            "options": ["Tips decrease as total bill increases", "Tips increase as total bill increases", "There's no relationship between tips and total bill"],
            "answer": 1,
            "explanation": "A positive slope indicates that as the total bill increases, the predicted tip amount also increases. This aligns with our intuition that people generally tip more for larger bills."
        },
        {
            "question": "What does an R² score of 0.7 mean?",
            "options": ["The model explains 70% of the variability in the data", "The model is 70% accurate", "70% of the predictions are correct"],
            "answer": 0,
            "explanation": "An R² score of 0.7 means that 70% of the variance in the dependent variable (tips) can be explained by the independent variable (total bill) in our model. It's a measure of how well the regression line approximates the real data points."
        },
        {
            "question": "What's the main purpose of the residual plot?",
            "options": ["To show the best fit line", "To identify outliers", "To check for patterns in prediction errors"],
            "answer": 2,
            "explanation": "The main purpose of a residual plot is to check for patterns in the prediction errors (residuals). Ideally, residuals should be randomly scattered around zero. Any visible patterns might indicate that a linear model is not appropriate or that there are other factors influencing the relationship."
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

Congratulations on exploring the Tip Predictor! Remember:

- 📊 Linear regression helps us understand and predict relationships between variables.
- 🧮 The best fit line minimizes the overall prediction error.
- 🚀 Always consider the context and limitations of your model when interpreting results.

Keep exploring and happy predicting!
""")
