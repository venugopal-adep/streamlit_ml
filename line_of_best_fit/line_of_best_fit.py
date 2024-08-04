import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set page config
st.set_page_config(page_title="Tip Predictor: Best Fit Line Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("💰 Tip Predictor: Best Fit Line Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore the relationship between total bill amounts and tips using linear regression!")

# Helper functions
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
    fig.update_layout(template="plotly_white", xaxis_title="Total Bill ($)", yaxis_title="Tip Amount ($)")
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Tips', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Plot", template="plotly_white", xaxis_title="Predicted Tips ($)", yaxis_title="Residuals ($)")
    return fig

# Sidebar
st.sidebar.header("Data Generation")
if st.sidebar.button("🔄 Generate New Data"):
    st.session_state.data = generate_data()
    st.session_state.predictions, st.session_state.model = train_and_predict(st.session_state.data)
    st.sidebar.success("New data generated and model retrained!")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_data()
    st.session_state.predictions, st.session_state.model = train_and_predict(st.session_state.data)
if 'show_best_fit' not in st.session_state:
    st.session_state.show_best_fit = True

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🔬 Explore", "📊 Model", "📈 Metrics", "🧠 Quiz"])

with tab1:
    st.header("Understanding Linear Regression")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Linear Regression?</h3>
    <p>Linear regression is a statistical method used to model the relationship between two variables by fitting a linear equation to observed data. It helps us understand and predict how one variable changes with respect to another.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts</h3>
    <ul>
        <li><strong>Best Fit Line:</strong> The line that minimizes the sum of squared differences between observed and predicted values.</li>
        <li><strong>Slope:</strong> Represents the change in the dependent variable (tips) for a one-unit change in the independent variable (total bill).</li>
        <li><strong>Intercept:</strong> The expected value of tips when the total bill is zero (often not meaningful in real-world contexts).</li>
        <li><strong>R² Score:</strong> Measures the proportion of variance in the dependent variable that is predictable from the independent variable.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Linear Regression?</h3>
    <ul>
        <li><span class="highlight">Prediction:</span> Estimate unknown values of one variable based on another.</li>
        <li><span class="highlight">Relationship Understanding:</span> Quantify how variables are related to each other.</li>
        <li><span class="highlight">Trend Analysis:</span> Identify and extrapolate trends in data.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("🔬 Data Explorer")
    
    st.session_state.show_best_fit = st.checkbox("Show Best Fit Line", value=st.session_state.show_best_fit)
    
    fig = plot_data(st.session_state.data, st.session_state.predictions, st.session_state.show_best_fit)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Raw Data"):
        st.write(st.session_state.data)

with tab3:
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

with tab4:
    st.header("📈 Model Metrics")
    
    y_true = st.session_state.data['tips']
    y_pred = st.session_state.predictions
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    
    st.subheader("Residual Plot")
    fig_residuals = plot_residuals(y_true, y_pred)
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.write("""
    The residual plot helps us check for patterns in prediction errors. Ideally, residuals should be randomly scattered around zero. 
    Any visible patterns might indicate that a linear model is not appropriate or that there are other factors influencing the relationship.
    """)

with tab5:
    st.header("🧠 Test Your Knowledge")

    questions = [
        {
            "question": "What does the slope in our model represent?",
            "options": ["The average tip amount", "How much the tip changes for each dollar increase in the bill", "The total bill amount"],
            "correct": 1,
            "explanation": "The slope represents how much the tip is expected to change for each dollar increase in the total bill."
        },
        {
            "question": "What does a higher R² score indicate?",
            "options": ["A weaker relationship between variables", "A stronger relationship between variables", "No relationship between variables"],
            "correct": 1,
            "explanation": "A higher R² score indicates a stronger relationship between the variables, meaning the model explains more of the variability in the data."
        },
        {
            "question": "What's the purpose of the best fit line?",
            "options": ["To connect all data points", "To minimize the overall prediction error", "To make the graph look better"],
            "correct": 1,
            "explanation": "The best fit line is chosen to minimize the overall prediction error, providing the best linear approximation of the relationship between variables."
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
st.sidebar.info("This app demonstrates linear regression for tip prediction. Generate new data, explore the model, and test your understanding!")
