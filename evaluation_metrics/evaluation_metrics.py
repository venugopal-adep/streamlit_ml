import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to generate synthetic data
def generate_data(n, noise_level):
    np.random.seed(42)
    X = np.linspace(0, 10, n)
    Y = 2 * X + 3 + np.random.normal(0, noise_level, n)
    return pd.DataFrame({'X': X, 'Y': Y})

# Function to fit the model and calculate metrics
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

# Sidebar controls
st.sidebar.header('Data Controls')
n_points = st.sidebar.slider('Number of Data Points', min_value=50, max_value=500, value=100, step=50)
noise_level = st.sidebar.slider('Noise Level', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
data = generate_data(n_points, noise_level)

# Fit model and calculate metrics
Y_pred, r2, adj_r2, mae, rmse, model = fit_and_evaluate(data)

# Display the metrics
st.write("## Regression Metrics : R2 score, MAE, MSE")
st.write("**Developed by : Venugopal Adep**")
st.write(f"**R-squared:** {r2:.4f}")
st.write(f"**Adjusted R-squared:** {adj_r2:.4f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Root Mean Square Error (RMSE):** {rmse:.4f}")

# Plotting the results
fig = px.scatter(data, x='X', y='Y', title="Linear Regression Fit")
fig.add_scatter(x=data['X'], y=Y_pred, mode='lines', name='Regression Line')
st.plotly_chart(fig)

# Explanation and educational content
st.write("""
## Evaluation Methods in Regression
- **R-squared:** Measure of the percentage of the variance in the dependent variable that is predictable from the independent variable(s).
- **Adjusted R-squared:** Adjusts the R-Squared value for the number of predictors in a regression model. Useful for multiple regression.
- **Mean Absolute Error (MAE):** Represents average error.
- **Root Mean Square Error (RMSE):** Also measures the average error, more sensitive to outliers than MAE.
""")
