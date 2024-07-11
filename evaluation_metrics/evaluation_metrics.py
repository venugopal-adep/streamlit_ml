import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(page_title="Regression Metrics Explorer", layout="wide")

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
st.title("📊 Regression Metrics Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Regression Metrics Explorer! Dive into the world of linear regression
and understand how different metrics evaluate model performance. Explore the impact
of data points and noise on regression results and metrics.
""")

# Functions
@st.cache_data
def generate_data(n, noise_level):
    np.random.seed(42)
    X = np.linspace(0, 10, n)
    Y = 2 * X + 3 + np.random.normal(0, noise_level, n)
    return pd.DataFrame({'X': X, 'Y': Y})

def fit_and_evaluate(data):
    model = LinearRegression()
    X = data[['X']]
    Y = data['Y']
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    r2 = r2_score(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    adj_r2 = 1 - (1-r2) * (len(Y)-1) / (len(Y)-X.shape[1]-1)
    
    return Y_pred, r2, adj_r2, mae, rmse, model

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "📈 Metrics", "🔬 Explore", "📚 Learn"])

with tab1:
    st.header("📊 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Data Controls")
        n_points = st.slider('Number of Data Points', min_value=50, max_value=500, value=100, step=50)
        noise_level = st.slider('Noise Level', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        data = generate_data(n_points, noise_level)
        Y_pred, r2, adj_r2, mae, rmse, model = fit_and_evaluate(data)
        
        st.subheader("Regression Equation")
        st.latex(f"Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}")
        
        st.subheader("Model Metrics")
        st.metric("R-squared", f"{r2:.4f}")
        st.metric("Adjusted R-squared", f"{adj_r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    
    with col1:
        fig = px.scatter(data, x='X', y='Y', title="Linear Regression Fit")
        fig.add_scatter(x=data['X'], y=Y_pred, mode='lines', name='Regression Line')
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Data"):
        st.write(data)

with tab2:
    st.header("📈 Metrics Interpretation")
    
    st.subheader("R-squared (R²)")
    st.write(f"""
    Current value: {r2:.4f}
    
    R-squared is a statistical measure that represents the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). It ranges from 0 to 1, where:
    - 0 indicates that the model explains none of the variability of the response data around its mean.
    - 1 indicates that the model explains all the variability.
    
    In this case, {r2:.2%} of the variance in Y can be explained by X.
    """)
    
    st.subheader("Adjusted R-squared")
    st.write(f"""
    Current value: {adj_r2:.4f}
    
    Adjusted R-squared is a modified version of R-squared that adjusts for the number of predictors in a model.
    It increases only if the new term improves the model more than would be expected by chance.
    It's particularly useful when comparing models with different numbers of predictors.
    """)
    
    st.subheader("Mean Absolute Error (MAE)")
    st.write(f"""
    Current value: {mae:.4f}
    
    MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
    It's the average over the test sample of the absolute differences between prediction and actual observation.
    In this case, on average, our predictions are off by {mae:.2f} units.
    """)
    
    st.subheader("Root Mean Square Error (RMSE)")
    st.write(f"""
    Current value: {rmse:.4f}
    
    RMSE is the square root of the average of squared differences between prediction and actual observation.
    It gives a relatively high weight to large errors, making it useful when large errors are particularly undesirable.
    Our model's RMSE of {rmse:.2f} indicates the standard deviation of the residuals (prediction errors).
    """)

with tab3:
    st.header("🔬 Explore Relationships")
    
    st.subheader("Impact of Noise on Metrics")
    noise_range = np.linspace(0.1, 10, 100)
    metrics = [fit_and_evaluate(generate_data(100, noise))[:4] for noise in noise_range]
    metrics_df = pd.DataFrame({
        'Noise': noise_range,
        'R-squared': [m[1] for m in metrics],
        'Adjusted R-squared': [m[2] for m in metrics],
        'MAE': [m[3] for m in metrics]
    })
    
    fig_metrics = px.line(metrics_df, x='Noise', y=['R-squared', 'Adjusted R-squared', 'MAE'],
                          title="Impact of Noise on Metrics")
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.write("""
    This plot shows how different levels of noise in the data affect our regression metrics.
    As noise increases:
    - R-squared and Adjusted R-squared decrease, indicating a poorer fit.
    - MAE increases, showing larger average prediction errors.
    """)

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Key Concepts in Regression Analysis")
    st.write("""
    1. **Linear Regression**: A statistical method for modeling the relationship between a dependent variable and one or more independent variables.
    
    2. **Residuals**: The differences between the observed values and the predicted values from the regression model.
    
    3. **Overfitting**: When a model learns the training data too well, including its noise and fluctuations, leading to poor generalization to new data.
    
    4. **Underfitting**: When a model is too simple to capture the underlying structure of the data, leading to poor performance on both training and new data.
    
    5. **Bias-Variance Tradeoff**: The property of a model that the variance of the parameter estimates across samples can be reduced by increasing the bias in the estimated parameters.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What does a R-squared value of 0.75 indicate?",
            "options": ["The model explains 75% of the variability in the data", "The model is 75% accurate", "75% of the predictions are correct"],
            "answer": 0,
            "explanation": "An R-squared value of 0.75 indicates that 75% of the variance in the dependent variable can be explained by the independent variable(s) in the model."
        },
        {
            "question": "When is Adjusted R-squared particularly useful?",
            "options": ["When dealing with time series data", "When comparing models with different numbers of predictors", "When the data is normally distributed"],
            "answer": 1,
            "explanation": "Adjusted R-squared is particularly useful when comparing models with different numbers of predictors. It adjusts for the number of predictors in the model, penalizing the addition of extraneous predictors."
        },
        {
            "question": "Which metric is more sensitive to outliers?",
            "options": ["MAE", "RMSE", "R-squared"],
            "answer": 1,
            "explanation": "RMSE (Root Mean Square Error) is more sensitive to outliers than MAE. This is because RMSE squares the errors before averaging them, which gives more weight to large errors."
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

Congratulations on exploring Regression Metrics! Remember:

- 📊 Different metrics provide different insights into model performance.
- 🔍 Consider multiple metrics when evaluating your regression model.
- 📈 The choice of metric depends on your specific problem and goals.

Keep exploring and refining your understanding of regression analysis!
""")
