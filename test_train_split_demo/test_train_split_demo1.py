import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Train-Test Split Regressor", page_icon="📈", layout="wide")
st.title("Train-Test Split Regressor")
st.write('**Developed by : Venugopal Adep**')

@st.cache_data
def generate_data(n_samples, n_features, noise):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

def plot_data(X, y, X_train, X_test, y_train, y_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', name='All Data'))
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Training Data'))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Testing Data'))
    fig.update_layout(
        title="Regression Data Visualization",
        xaxis_title="Feature",
        yaxis_title="Target",
        legend_title="Data Split"
    )
    return fig

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_predictions(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Training Data'))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Testing Data'))
    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_pred_train, mode='lines', name='Training Predictions'))
    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_pred_test, mode='lines', name='Testing Predictions'))
    fig.update_layout(
        title="Regression Predictions",
        xaxis_title="Feature",
        yaxis_title="Target",
        legend_title="Data and Predictions"
    )
    return fig

# Sidebar
st.sidebar.title("Parameters")
n_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=500, value=200, step=50)
n_features = st.sidebar.slider("Number of Features", min_value=1, max_value=10, value=1, step=1)
noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=100.0, value=30.0, step=10.0)
test_size = st.sidebar.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

# Generate data
X, y = generate_data(n_samples, n_features, noise)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Plot data
fig = plot_data(X, y, X_train, X_test, y_train, y_test)
st.plotly_chart(fig, use_container_width=True)

# Train model
model = train_model(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot predictions
fig_pred = plot_predictions(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)
st.plotly_chart(fig_pred, use_container_width=True)

# Evaluate model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Display results
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Training Set Size", value=len(X_train))
    st.metric(label="Training MSE", value=f"{train_mse:.2f}")
    st.metric(label="Training R-squared", value=f"{train_r2:.2f}")
with col2:
    st.metric(label="Testing Set Size", value=len(X_test))
    st.metric(label="Testing MSE", value=f"{test_mse:.2f}")
    st.metric(label="Testing R-squared", value=f"{test_r2:.2f}")

st.markdown("""
## Explanation

This interactive application demonstrates the concept of train-test split in a regression problem. Here's how it works:

1. The application generates a synthetic regression dataset using the `make_regression` function from scikit-learn. You can adjust the number of samples, number of features, and noise level using the sliders in the sidebar.

2. The generated data is split into training and testing sets using the `train_test_split` function. You can control the size of the test set using the slider in the sidebar.

3. The data points are visualized using a scatter plot, with different colors representing the training and testing data.

4. A linear regression model is trained on the training data and used to make predictions on both the training and testing sets.

5. The predictions are plotted along with the actual data points, allowing you to visually assess the model's performance.

6. The mean squared error (MSE) and R-squared (R2) metrics are calculated for both the training and testing sets. These metrics provide a quantitative measure of the model's performance.

7. The training and testing set sizes, along with their respective MSE and R-squared values, are displayed using Streamlit's `metric` function.

This application highlights the importance of train-test split in evaluating the performance of regression models. By adjusting the various parameters and observing the model's predictions and evaluation metrics, you can gain insights into how well the model generalizes to unseen data.

Feel free to experiment with different parameter values and see how they affect the model's performance!
""")