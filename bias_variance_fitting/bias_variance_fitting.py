import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def generate_data(n_points):
    np.random.seed(42)
    X = np.linspace(0, 10, n_points)
    Y = np.sin(X) + np.random.normal(0, 0.5, n_points)  # Sine wave with noise
    X = X.reshape(-1, 1)
    return X, Y

def fit_models(X, Y, degrees):
    models = {}
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, Y)
        models[degree] = model
    return models

def plot_models(X, Y, models, X_test, Y_test):
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    fig = px.scatter(x=X.squeeze(), y=Y, labels={'x': 'X', 'y': 'Y'}, title="Model Fitting Demonstrations")
    fig.add_scatter(x=X_test.squeeze(), y=Y_test, mode='markers', name='Test Data')

    for degree, model in models.items():
        Y_plot = model.predict(X_plot)
        fig.add_scatter(x=X_plot.squeeze(), y=Y_plot, mode='lines', name=f'Degree {degree}')

    return fig

# Streamlit interface
st.write("## Bias-Variance - Fitting - Demo")
st.write("**Developed by : Venugopal Adep**")
n_points = st.sidebar.slider("Number of Data Points", 30, 150, 50)
degrees = st.sidebar.multiselect("Degrees of Polynomial", [1, 2, 3, 5, 10, 20], default=[1, 3, 10])

# Data generation
X, Y = generate_data(n_points)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fitting models
models = fit_models(X_train, Y_train, degrees)

# Plotting
fig = plot_models(X_train, Y_train, models, X_test, Y_test)
st.plotly_chart(fig)

# Displaying metrics
for degree, model in models.items():
    train_score = r2_score(Y_train, model.predict(X_train))
    test_score = r2_score(Y_test, model.predict(X_test))
    st.write(f"Degree {degree}: Train R² = {train_score:.2f}, Test R² = {test_score:.2f}")

st.write("""
### Insights
- **Lower Degree Models**: May underfit the data (high bias, low variance).
- **Higher Degree Models**: May overfit the data (low bias, high variance).
- **Optimal Degree**: Balances bias and variance to generalize well on unseen data.
""")
