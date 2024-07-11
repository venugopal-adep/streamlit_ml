import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Linear Regression: Residual Error Explorer", layout="wide")

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
st.title("📏 Linear Regression: Residual Error Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Linear Regression Residual Error Explorer! Dive into the world of linear regression
and understand how residuals play a crucial role in model evaluation. Explore different datasets,
visualize the regression line, and analyze the residuals to gain insights into model performance.
""")

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

# Functions
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

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "🔬 Residuals", "📈 Metrics", "📚 Learn"])

with tab1:
    st.header("📊 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Dataset selection
        dataset_choice = st.selectbox("Select Dataset", list(datasets.keys()))
        data = datasets[dataset_choice]
        
        # Train model and make predictions
        model = train_model(data)
        data['Predicted Weight'] = model.predict(data[['Height']])
        data['Residual'] = data['Weight'] - data['Predicted Weight']
        data['Residual^2'] = data['Residual'] ** 2
        
        # Regression Equation
        st.subheader("Regression Equation")
        st.latex(f"\\hat{{Y}} = {model.intercept_:.2f} + {model.coef_[0]:.4f}X")
        
        # Calculate metrics
        mae, mse, r2 = calculate_metrics(data, data['Predicted Weight'])
        
        # Display metrics
        st.subheader("Model Metrics")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
    
    with col1:
        # Model Visualization
        fig = plot_data_with_residuals(data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Table (collapsible)
    with st.expander("View Data Table"):
        st.write(data)

with tab2:
    st.header("🔬 Residual Analysis")
    
    st.subheader("Residual Plot")
    fig_residuals = px.scatter(data, x='Predicted Weight', y='Residual', 
                               title='Residual Plot',
                               labels={'Predicted Weight': 'Predicted Weight (lbs)', 'Residual': 'Residual (lbs)'})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.subheader("Residual Distribution")
    fig_hist = px.histogram(data, x='Residual', title='Distribution of Residuals')
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.header("📈 Model Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")
    
    st.subheader("Interpretation")
    st.write(f"""
    - **MAE**: On average, our predictions are off by {mae:.2f} lbs.
    - **MSE**: The average squared difference between predicted and actual weights is {mse:.2f} lbs².
    - **R²**: {r2:.2%} of the variance in weight can be explained by height in this model.
    """)

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Understanding Linear Regression and Residuals")
    st.write("""
    Linear regression is a statistical method for modeling the relationship between a dependent variable (y) and one or more independent variables (x). In this case:
    
    - **x (independent variable)**: Height
    - **y (dependent variable)**: Weight
    - **Regression Line**: The line that best fits the data points
    - **Residuals**: The vertical distance between each data point and the regression line
    
    Key points to remember:
    1. The regression line is obtained by minimizing the sum of squared residuals.
    2. Residuals represent the error in our predictions.
    3. Analyzing residuals helps us assess the model's performance and assumptions.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What does a residual represent in this context?",
            "options": ["The predicted weight", "The difference between actual and predicted weight", "The height of a person"],
            "answer": 1,
            "explanation": "A residual is the difference between the actual weight and the predicted weight for each data point. It represents how far off our prediction is from the actual value."
        },
        {
            "question": "What does a high R² score indicate?",
            "options": ["Poor model fit", "Strong linear relationship", "No relationship between variables"],
            "answer": 1,
            "explanation": "A high R² score (close to 1) indicates a strong linear relationship between the variables. It suggests that a large portion of the variability in weight can be explained by height in our model."
        },
        {
            "question": "In an ideal residual plot, what pattern should we see?",
            "options": ["A clear trend", "Random scatter around zero", "All positive residuals"],
            "answer": 1,
            "explanation": "In an ideal residual plot, we should see a random scatter of points around the zero line. This suggests that our model's errors are random and not systematically biased in any direction."
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

Congratulations on exploring Linear Regression and Residual Analysis! Remember:

- 📊 The regression line helps us understand the relationship between variables.
- 🔍 Residuals are crucial for assessing model performance and assumptions.
- 📈 Metrics like MAE, MSE, and R² provide quantitative measures of model fit.

Keep exploring and refining your understanding of statistical modeling!
""")
