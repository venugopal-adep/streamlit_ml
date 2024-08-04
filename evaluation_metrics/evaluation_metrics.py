import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="Regression Metrics Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("📊 Regression Metrics Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the key metrics used to evaluate regression models!")

# Helper functions
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

def plot_regression(data, Y_pred):
    fig = px.scatter(data, x='X', y='Y', title="Linear Regression Fit")
    fig.add_scatter(x=data['X'], y=Y_pred, mode='lines', name='Regression Line')
    fig.update_layout(template="plotly_white")
    return fig

def plot_metrics_vs_noise(noise_range, metrics_df):
    fig = px.line(metrics_df, x='Noise', y=['R-squared', 'Adjusted R-squared', 'MAE'],
                  title="Impact of Noise on Metrics")
    fig.update_layout(template="plotly_white")
    return fig

# Sidebar
st.sidebar.header("Data Configuration")
n_points = st.sidebar.slider('Number of Data Points', min_value=50, max_value=500, value=100, step=50)
noise_level = st.sidebar.slider('Noise Level', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Generate data
if 'data' not in st.session_state or st.sidebar.button("Generate New Data"):
    st.session_state.data = generate_data(n_points, noise_level)
    st.session_state.Y_pred, st.session_state.r2, st.session_state.adj_r2, st.session_state.mae, st.session_state.rmse, st.session_state.model = fit_and_evaluate(st.session_state.data)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Model", "📈 Metrics", "🧠 Quiz"])

with tab1:
    st.header("Understanding Regression Metrics")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What are Regression Metrics?</h3>
    <p>Regression metrics are statistical measures used to evaluate the performance of a regression model. They help us understand how well the model fits the data and makes predictions. Key metrics include:</p>
    <ul>
        <li><strong>R-squared (R²):</strong> Measures the proportion of variance in the dependent variable explained by the independent variables.</li>
        <li><strong>Adjusted R-squared:</strong> A modified version of R² that adjusts for the number of predictors in the model.</li>
        <li><strong>Mean Absolute Error (MAE):</strong> The average of the absolute differences between predictions and actual values.</li>
        <li><strong>Root Mean Square Error (RMSE):</strong> The square root of the average of squared differences between predictions and actual values.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why are these Metrics Important?</h3>
    <ul>
        <li><span class="highlight">Model Comparison:</span> Help in comparing different models to choose the best one.</li>
        <li><span class="highlight">Performance Evaluation:</span> Provide insights into how well the model is performing.</li>
        <li><span class="highlight">Error Understanding:</span> Give different perspectives on the types of errors in the model's predictions.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_regression(st.session_state.data, st.session_state.Y_pred)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regression Equation")
        st.latex(f"Y = {st.session_state.model.coef_[0]:.2f}X + {st.session_state.model.intercept_:.2f}")
        
        st.subheader("Model Metrics")
        st.metric("R-squared", f"{st.session_state.r2:.4f}")
        st.metric("Adjusted R-squared", f"{st.session_state.adj_r2:.4f}")
        st.metric("MAE", f"{st.session_state.mae:.4f}")
        st.metric("RMSE", f"{st.session_state.rmse:.4f}")
    
    with st.expander("View Data"):
        st.write(st.session_state.data)

with tab3:
    st.header("📈 Metrics Interpretation")
    
    st.subheader("R-squared (R²)")
    st.write(f"""
    Current value: {st.session_state.r2:.4f}
    
    R-squared represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    
    In this case, {st.session_state.r2:.2%} of the variance in Y can be explained by X.
    """)
    
    st.subheader("Adjusted R-squared")
    st.write(f"""
    Current value: {st.session_state.adj_r2:.4f}
    
    Adjusted R-squared modifies R-squared by accounting for the number of predictors in the model. It's useful when comparing models with different numbers of predictors.
    """)
    
    st.subheader("Mean Absolute Error (MAE)")
    st.write(f"""
    Current value: {st.session_state.mae:.4f}
    
    MAE represents the average magnitude of the errors in a set of predictions, without considering their direction.
    On average, our predictions are off by {st.session_state.mae:.2f} units.
    """)
    
    st.subheader("Root Mean Square Error (RMSE)")
    st.write(f"""
    Current value: {st.session_state.rmse:.4f}
    
    RMSE is the square root of the average of squared differences between prediction and actual observation.
    It gives a relatively high weight to large errors.
    """)
    
    st.subheader("Impact of Noise on Metrics")
    noise_range = np.linspace(0.1, 10, 100)
    metrics = [fit_and_evaluate(generate_data(100, noise))[:4] for noise in noise_range]
    metrics_df = pd.DataFrame({
        'Noise': noise_range,
        'R-squared': [m[1] for m in metrics],
        'Adjusted R-squared': [m[2] for m in metrics],
        'MAE': [m[3] for m in metrics]
    })
    
    fig_metrics = plot_metrics_vs_noise(noise_range, metrics_df)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.write("""
    This plot shows how different levels of noise in the data affect our regression metrics.
    As noise increases:
    - R-squared and Adjusted R-squared decrease, indicating a poorer fit.
    - MAE increases, showing larger average prediction errors.
    """)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does R-squared measure?",
            "options": ["The proportion of variance explained by the model", "The average error of the model", "The slope of the regression line", "The number of predictors in the model"],
            "correct": 0,
            "explanation": "R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variable(s)."
        },
        {
            "question": "Which metric is not affected by the scale of the data?",
            "options": ["MAE", "RMSE", "R-squared", "All of the above"],
            "correct": 2,
            "explanation": "R-squared is not affected by the scale of the data because it represents a proportion of explained variance."
        },
        {
            "question": "What happens to the regression metrics as noise in the data increases?",
            "options": ["R-squared increases", "MAE decreases", "RMSE decreases", "R-squared decreases"],
            "correct": 3,
            "explanation": "As noise in the data increases, R-squared typically decreases because the model's ability to explain the variance in the data decreases."
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
st.sidebar.info("This app demonstrates regression metrics. Adjust the settings, generate new data, and explore the different tabs to learn more!")
