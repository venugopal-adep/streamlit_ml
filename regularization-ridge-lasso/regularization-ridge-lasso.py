import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate synthetic data
def generate_data(n_samples=100, n_features=10, noise=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, y, coef = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, coef=True, random_state=seed)
    return X, y, coef

# Fit model
def fit_model(X_train, y_train, alpha, model_type='lasso'):
    if model_type == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

# Plot coefficients
def plot_coefficients(coefs, true_coefs, model_type):
    df = pd.DataFrame({
        'True Coefficients': true_coefs,
        'Estimated Coefficients': coefs
    })
    fig = px.bar(df, barmode='group', title=f'{model_type.capitalize()} Regression Coefficients Comparison')
    return fig

# Streamlit interface
st.write("## Regularization Techniques - Lasso & Ridge")
st.write("**Developed by: Venugopal Adep**")

# Sidebar for input parameters
st.sidebar.write("**Model Configuration**")
model_type = st.sidebar.selectbox("Select Regularization Type", ['lasso', 'ridge'])
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 1.0, 0.1, 0.01)

# Button to regenerate data
if st.sidebar.button('Regenerate Data'):
    new_seed = np.random.randint(10000)  # Generate a new random seed for data regeneration
    st.session_state['X'], st.session_state['y'], st.session_state['true_coefs'] = generate_data(seed=new_seed)

# Initialize data if not already in session state
if 'X' not in st.session_state:
    st.session_state['X'], st.session_state['y'], st.session_state['true_coefs'] = generate_data()

# Data display and model fitting
X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], st.session_state['y'], test_size=0.2, random_state=42)
model = fit_model(X_train, y_train, alpha, model_type)

# Display the specific regularization equation
if model_type == 'lasso':
    st.latex(r"Lasso: \min_{\beta} \left\{ \frac{1}{2n} \left\| y - X\beta \right\|^2 + \alpha \left\| \beta \right\|_1 \right\}")
else:
    st.latex(r"Ridge: \min_{\beta} \left\{ \frac{1}{2n} \left\| y - X\beta \right\|^2 + \alpha \left\| \beta \right\|^2_2 \right\}")

# Data and plotting
df = pd.DataFrame(st.session_state['X'], columns=[f'Feature_{i+1}' for i in range(st.session_state['X'].shape[1])])
df['Target'] = st.session_state['y']
st.write("### Sample Data (First 5 Rows)")
st.dataframe(df.head())
predicted_coefs = model.coef_
test_mse = mean_squared_error(y_test, model.predict(X_test))
fig = plot_coefficients(predicted_coefs, st.session_state['true_coefs'], model_type)
st.plotly_chart(fig)
st.write(f"Test Mean Squared Error: {test_mse:.2f}")

st.write("### Insights:")
st.write("""
- **Coefficients**: Notice how the coefficients shrink towards zero as alpha increases, especially with Lasso.
- **MSE**: Observe how the prediction error changes with different alpha values.
""")
