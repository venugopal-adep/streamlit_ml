import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic data based on the regression equation
def generate_data(num_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    height = np.linspace(60, 75, num_samples)  # Heights ranging from 60 to 75 inches
    weight = -266.53 + 6.1376 * height + np.random.normal(0, 5, num_samples)  # Equation with added noise
    return pd.DataFrame({'Height': height, 'Weight': weight})

# Function to train model
def train_model(data):
    model = LinearRegression()
    model.fit(data[['Height']], data['Weight'])
    return model

# Check if 'data' and 'model' are already in state, else initialize
if 'data' not in st.session_state or 'model' not in st.session_state:
    st.session_state['data'] = generate_data(100)
    st.session_state['model'] = train_model(st.session_state['data'])

# Sidebar for input parameters and actions
st.sidebar.header('Input Parameters and Actions')

# Button to regenerate data
if st.sidebar.button('Regenerate Data'):
    new_seed = np.random.randint(10000)  # Generate a new random seed
    st.session_state['data'] = generate_data(100, seed=new_seed)
    st.session_state['model'] = train_model(st.session_state['data'])

# Display Regression Equation
model = st.session_state['model']
st.write("## Linear Regression Demonstration")
st.write("**Developed by : Venugopal Adep**")
st.latex(f"\\hat{{Y}} = {model.intercept_:.2f} + {model.coef_[0]:.4f}X")
st.write("Where X is the height in inches and Ŷ is the predicted weight in lbs.")

# Plotting function with an additional point for prediction
def plot_regression_line(data, model, prediction_point=None):
    fig = px.scatter(data, x='Height', y='Weight', title='Height vs. Weight Regression Analysis',
                     labels={'Height': 'Height (inches)', 'Weight': 'Weight (lbs)'}, opacity=0.7)
    fig.add_scatter(x=data['Height'], y=model.predict(data[['Height']]), mode='lines', name='Regression Line', line=dict(color='blue'))
    if prediction_point:
        fig.add_trace(px.scatter(x=[prediction_point[0]], y=[prediction_point[1]], labels={'x': 'Predicted Height', 'y': 'Predicted Weight'}).data[0])
        fig.data[-1].update(marker=dict(color='red', size=10), name='Prediction Point')
    return fig

# User input and prediction logic in sidebar
input_height = st.sidebar.slider('Height (inches)', int(st.session_state['data']['Height'].min()), int(st.session_state['data']['Height'].max()), 68)
predicted_weight = model.predict([[input_height]])[0]
st.sidebar.write(f'Predicted Weight: {predicted_weight:.2f} lbs')

# Calculate metrics in sidebar
predictions = model.predict(st.session_state['data'][['Height']])
mae = mean_absolute_error(st.session_state['data']['Weight'], predictions)
mse = mean_squared_error(st.session_state['data']['Weight'], predictions)
r2 = r2_score(st.session_state['data']['Weight'], predictions)
st.sidebar.write(f"### Model Metrics")
st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.sidebar.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.sidebar.write(f"R² Score: {r2:.2f}")

# Plot with the predicted point marked
fig = plot_regression_line(st.session_state['data'], model, [input_height, predicted_weight])
st.plotly_chart(fig)

# Explanation of Intercept
st.write("""
### Understanding the Intercept
- The intercept in this regression equation is the theoretical weight when height is zero, which is not practical.
- Extrapolating beyond the observed range of data can lead to unrealistic predictions.
- Always consider the range of your data before applying the model to make predictions.
""")
