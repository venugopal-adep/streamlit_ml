import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Linear Regression Residual Error Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("📏 Linear Regression: Residual Error Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the importance of residuals in linear regression analysis!")

# Define datasets
datasets = {
    'Dataset 1': pd.DataFrame({
        'Height': [63, 64, 66, 69, 69, 71, 71, 72, 73, 75],
        'Weight': [127, 121, 142, 157, 162, 156, 169, 165, 181, 208]
    }),
    'Dataset 2': pd.DataFrame({
        'Height': [62, 65, 67, 70, 72, 74, 76, 78, 80, 82],
        'Weight': [130, 135, 145, 160, 175, 190, 205, 220, 235, 250]
    }),
    'Dataset 3': pd.DataFrame({
        'Height': [60, 62, 63, 66, 68, 69, 72, 74, 76, 78],
        'Weight': [110, 115, 120, 130, 140, 150, 160, 170, 180, 190]
    }),
    'Dataset 4': pd.DataFrame({
        'Height': [58, 59, 61, 63, 65, 67, 67, 71, 73, 77],
        'Weight': [95, 100, 102, 112, 123, 135, 145, 157, 165, 172]
    })
}

# Helper functions
def train_model(data):
    model = LinearRegression()
    model.fit(data[['Height']], data['Weight'])
    return model

def calculate_metrics(data, predictions):
    mae = mean_absolute_error(data['Weight'], predictions)
    mse = mean_squared_error(data['Weight'], predictions)
    r2 = r2_score(data['Weight'], predictions)
    return mae, mse, r2

def plot_data_with_residuals(data):
    fig = px.scatter(data, x='Height', y='Weight', title='Height vs. Weight Regression Analysis',
                     labels={'Height': 'Height (inches)', 'Weight': 'Weight (lbs)'}, opacity=0.7)
    fig.add_scatter(x=data['Height'], y=data['Predicted Weight'], mode='lines', name='Regression Line')
    for i, row in data.iterrows():
        fig.add_shape(type='line', x0=row['Height'], y0=row['Weight'], x1=row['Height'], y1=row['Predicted Weight'],
                      line=dict(color='red', width=1))
    fig.update_layout(template="plotly_white")
    return fig

def plot_residuals(data):
    fig = px.scatter(data, x='Predicted Weight', y='Residual', 
                     title='Residual Plot',
                     labels={'Predicted Weight': 'Predicted Weight (lbs)', 'Residual': 'Residual (lbs)'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(template="plotly_white")
    return fig

def plot_residual_distribution(data):
    fig = px.histogram(data, x='Residual', title='Distribution of Residuals')
    fig.update_layout(template="plotly_white")
    return fig

# Sidebar
st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))

# Load and process data
data = datasets[dataset_choice]
model = train_model(data)
data['Predicted Weight'] = model.predict(data[['Height']])
data['Residual'] = data['Weight'] - data['Predicted Weight']
data['Residual^2'] = data['Residual'] ** 2
mae, mse, r2 = calculate_metrics(data, data['Predicted Weight'])

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Model", "🔬 Residuals", "🧠 Quiz"])

with tab1:
    st.header("Understanding Linear Regression and Residuals")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Linear Regression?</h3>
    <p>Linear regression is a statistical method for modeling the relationship between a dependent variable (y) and one or more independent variables (x). In this case:</p>
    <ul>
        <li><strong>x (independent variable):</strong> Height</li>
        <li><strong>y (dependent variable):</strong> Weight</li>
        <li><strong>Regression Line:</strong> The line that best fits the data points</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>What are Residuals?</h3>
    <p>Residuals are the differences between the observed values and the predicted values from the regression model. They represent the error in our predictions.</p>
    <ul>
        <li><span class="highlight">Residual = Actual Value - Predicted Value</span></li>
        <li>Analyzing residuals helps us assess the model's performance and assumptions.</li>
        <li>The regression line is obtained by minimizing the sum of squared residuals.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Metrics in Linear Regression</h3>
    <ul>
        <li><strong>Mean Absolute Error (MAE):</strong> Average of the absolute differences between predictions and actual values.</li>
        <li><strong>Mean Squared Error (MSE):</strong> Average of the squared differences between predictions and actual values.</li>
        <li><strong>R² Score:</strong> Proportion of the variance in the dependent variable that is predictable from the independent variable(s).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_data_with_residuals(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regression Equation")
        st.latex(f"\\hat{{Y}} = {model.intercept_:.2f} + {model.coef_[0]:.4f}X")
        
        st.subheader("Model Metrics")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
    
    with st.expander("View Data Table"):
        st.write(data)

with tab3:
    st.header("🔬 Residual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Residual Plot")
        fig_residuals = plot_residuals(data)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col2:
        st.subheader("Residual Distribution")
        fig_hist = plot_residual_distribution(data)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Interpretation")
    st.write("""
    - The Residual Plot shows the difference between actual and predicted values.
    - Ideally, residuals should be randomly scattered around the zero line.
    - The Residual Distribution should be approximately normal and centered around zero.
    """)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does a residual represent in linear regression?",
            "options": ["The predicted value", "The actual value", "The difference between actual and predicted value"],
            "correct": 2,
            "explanation": "A residual is the difference between the actual observed value and the predicted value from the regression model."
        },
        {
            "question": "What does a high R² score indicate?",
            "options": ["Poor model fit", "Strong relationship between variables", "No relationship between variables"],
            "correct": 1,
            "explanation": "A high R² score (close to 1) indicates a strong relationship between the variables, suggesting that a large portion of the variability in the dependent variable can be explained by the independent variable(s)."
        },
        {
            "question": "In an ideal residual plot, what pattern should we see?",
            "options": ["A clear upward trend", "A clear downward trend", "Random scatter around zero"],
            "correct": 2,
            "explanation": "In an ideal residual plot, we should see a random scatter of points around the zero line. This suggests that our model's errors are random and not systematically biased in any direction."
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
st.sidebar.info("This app explores linear regression and residual analysis. Select different datasets to see how the model and residuals change!")
