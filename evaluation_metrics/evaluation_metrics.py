import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="Regression Model Explorer", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f5f7ff;
}
.metric-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.formula-box {
    background-color: #f0f0f8;
    border-left: 5px solid #6c5ce7;
    padding: 10px;
    border-radius: 0 5px 5px 0;
    margin: 10px 0;
}
.metric-table {
    width: 100%;
    text-align: left;
    border-collapse: collapse;
}
.metric-table th, .metric-table td {
    padding: 12px 15px;
    border-bottom: 1px solid #e0e0e0;
}
.metric-table th {
    background-color: #6c5ce7;
    color: white;
}
.metric-table tr:nth-child(even) {
    background-color: #f8f9ff;
}
.interpretation {
    background-color: #e7f5fe;
    border-radius: 5px;
    padding: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Regression Model Explorer")
st.markdown("**Developed by: Venugopal Adep | Enhanced Version**")
st.markdown("Visualize your regression model and understand key performance metrics")

# Helper functions
@st.cache_data
def generate_data(n, noise_level, relationship_type='linear'):
    np.random.seed(42)
    X = np.linspace(0, 10, n)
    
    if relationship_type == 'linear':
        Y = 2 * X + 3 + np.random.normal(0, noise_level, n)
    elif relationship_type == 'quadratic':
        Y = 0.5 * X**2 - X + 5 + np.random.normal(0, noise_level, n)
    elif relationship_type == 'exponential':
        Y = 2 * np.exp(0.3 * X) + np.random.normal(0, noise_level, n)
    
    return pd.DataFrame({'X': X, 'Y': Y})

def fit_and_evaluate(data):
    model = LinearRegression()
    X = data[['X']]
    Y = data['Y']
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    adj_r2 = 1 - (1-r2) * (len(Y)-1) / (len(Y)-X.shape[1]-1)
    
    # Calculate residuals
    residuals = Y - Y_pred
    
    return Y_pred, r2, adj_r2, mae, mse, rmse, model, residuals

def plot_regression(data, Y_pred):
    fig = px.scatter(data, x='X', y='Y', title="Linear Regression Fit")
    fig.add_scatter(x=data['X'], y=Y_pred, mode='lines', name='Regression Line', 
                   line=dict(color='#6c5ce7', width=3))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(240, 240, 248, 0.6)',
        xaxis_title="X Variable",
        yaxis_title="Y Variable",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_residuals(data, residuals):
    residual_df = pd.DataFrame({'X': data['X'], 'Residuals': residuals})
    fig = px.scatter(residual_df, x='X', y='Residuals', title="Residual Plot")
    fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(240, 240, 248, 0.6)',
        xaxis_title="X Variable",
        yaxis_title="Residuals (Actual - Predicted)"
    )
    return fig

# Sidebar
st.sidebar.header("Data Configuration")
n_points = st.sidebar.slider('Number of Data Points', min_value=50, max_value=500, value=100, step=50)
noise_level = st.sidebar.slider('Noise Level', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
relationship = st.sidebar.selectbox('Relationship Type', 
                                   ['linear', 'quadratic', 'exponential'], 
                                   index=0)

# Generate data
if 'data' not in st.session_state or st.sidebar.button("Generate New Data"):
    st.session_state.data = generate_data(n_points, noise_level, relationship)
    st.session_state.Y_pred, st.session_state.r2, st.session_state.adj_r2, st.session_state.mae, st.session_state.mse, st.session_state.rmse, st.session_state.model, st.session_state.residuals = fit_and_evaluate(st.session_state.data)

# Main content
st.header("📊 Model Visualization and Metrics")

col1, col2 = st.columns([3, 2])

with col1:
    # Regression plot
    fig = plot_regression(st.session_state.data, st.session_state.Y_pred)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual plot
    fig_residuals = plot_residuals(st.session_state.data, st.session_state.residuals)
    st.plotly_chart(fig_residuals, use_container_width=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Regression Equation")
    st.latex(f"Y = {st.session_state.model.coef_[0]:.4f}X + {st.session_state.model.intercept_:.4f}")
    
    if relationship != 'linear':
        st.info(f"Note: You're fitting a linear model to {relationship} data. This might not be the best fit.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Regression Metrics Formulas")
    
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown("**R-squared (R²):**")
    st.latex(r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown("**Adjusted R-squared:**")
    st.latex(r"Adj\ R^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}")
    st.markdown("where n = number of samples, p = number of predictors")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown("**Mean Absolute Error (MAE):**")
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown("**Mean Squared Error (MSE):**")
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='formula-box'>", unsafe_allow_html=True)
    st.markdown("**Root Mean Squared Error (RMSE):**")
    st.latex(r"RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Metrics table
st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

metrics_data = {
    "Metric": ["R-squared (R²)", "Adjusted R-squared", "Mean Absolute Error (MAE)", 
               "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
    "Value": [f"{st.session_state.r2:.4f}", f"{st.session_state.adj_r2:.4f}", 
              f"{st.session_state.mae:.4f}", f"{st.session_state.mse:.4f}", 
              f"{st.session_state.rmse:.4f}"],
    "Interpretation": [
        f"{st.session_state.r2:.2%} of the variance in Y is explained by X. Higher is better (max 1.0).",
        f"Adjusts R² for the number of predictors. Useful for comparing models with different numbers of features.",
        f"On average, predictions are off by {st.session_state.mae:.4f} units. Lower is better.",
        f"Average of squared errors. Penalizes larger errors more than MAE. Lower is better.",
        f"Square root of MSE. In the same units as the dependent variable. Lower is better."
    ]
}

metrics_df = pd.DataFrame(metrics_data)
st.markdown(
    metrics_df.style.set_properties(**{'text-align': 'left'})
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#6c5ce7'), ('color', 'white')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f8f9ff')]},
    ])
    .to_html(), unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# Data viewer
with st.expander("View Data"):
    st.write("Data with Predictions and Metrics")
    prediction_df = st.session_state.data.copy()
    prediction_df['Predicted_Y'] = st.session_state.Y_pred
    prediction_df['Residuals'] = prediction_df['Y'] - prediction_df['Predicted_Y']
    prediction_df['Squared_Error'] = prediction_df['Residuals']**2
    prediction_df['Absolute_Error'] = np.abs(prediction_df['Residuals'])
    
    # Set a wider display for better visibility
    st.dataframe(prediction_df, width=1000, height=400)
    
    # Summary statistics
    st.write("Summary Statistics")
    summary_stats = {
        "Mean Absolute Error (MAE)": prediction_df['Absolute_Error'].mean(),
        "Mean Squared Error (MSE)": prediction_df['Squared_Error'].mean(),
        "Root Mean Squared Error (RMSE)": np.sqrt(prediction_df['Squared_Error'].mean()),
        "R-squared (R²)": st.session_state.r2,
        "Adjusted R-squared": st.session_state.adj_r2
    }
    st.dataframe(pd.DataFrame([summary_stats]), width=1000)


st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates regression model metrics. Adjust the settings and generate new data to see how different data characteristics affect model performance.")
