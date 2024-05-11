import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

def generate_data(num_samples, slope, intercept, noise):
    # Generate random data based on the input parameters
    X = np.random.rand(num_samples, 1)
    y = slope * X + intercept + noise * np.random.randn(num_samples, 1)
    return X, y

def train_model(X, y):
    # Train an OLS regression model
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    st.title("Ordinary Least Squares (OLS) Regression Demonstration")
    st.write("Developed by : **Venugopal Adep**")

    st.write("""
    ## Explanation of OLS Regression
    Ordinary Least Squares (OLS) is a type of linear regression that estimates the relationship between one or more independent variables and a dependent variable by minimizing the sum of the squares of the differences between the observed and predicted values.

    ### Mathematical Formula
    The model estimates parameters (slope and intercept) by solving:
    - Slope (β1) = Σ[(xi - mean(x)) * (yi - mean(y))] / Σ[(xi - mean(x))^2]
    - Intercept (β0) = mean(y) - β1 * mean(x)

    where (xi, yi) are data points, β1 is the slope, and β0 is the intercept.
    """)

    # Input parameters
    st.sidebar.header("Input Parameters")
    num_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, value=100, step=10)
    slope = st.sidebar.slider("Slope", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    intercept = st.sidebar.slider("Intercept", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
    noise = st.sidebar.slider("Noise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Generate data
    X, y = generate_data(num_samples, slope, intercept, noise)
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluation metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Display results
    st.subheader("Regression Results")
    st.write("Slope (Coefficient):", model.coef_[0][0])
    st.write("Intercept:", model.intercept_[0])
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared (R2):", r2)
    
    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(
        title="Ordinary Least Squares (OLS) Regression",
        xaxis_title="X",
        yaxis_title="y",
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
