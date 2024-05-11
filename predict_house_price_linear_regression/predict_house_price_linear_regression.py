import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Generate synthetic data
@st.cache_data
def generate_data(num_samples):
    np.random.seed(0)
    area = np.random.randint(1000, 7500, num_samples)
    rooms = np.random.randint(2, 10, num_samples)
    kitchens = np.random.randint(1, 4, num_samples)
    price = area * 200 + rooms * 25000 + kitchens * 5000 + np.random.randint(50000, 100000, num_samples)
    data = pd.DataFrame({
        'Area': area,
        'Rooms': rooms,
        'Kitchens': kitchens,
        'Price': price
    })
    return data

data = generate_data(300)

# Build the regression model
model = LinearRegression()
model.fit(data[['Area', 'Rooms', 'Kitchens']], data['Price'])

# Predict function
def predict_price(area, rooms, kitchens):
    return model.predict(np.array([[area, rooms, kitchens]]))[0]

# Display Regression Equation
st.write("## Predicting the House price using Regression")
st.write("**Developed by : Venugopal Adep**")
coefficients = model.coef_
intercept = model.intercept_
st.latex(f"Price = {intercept:.2f} + ({coefficients[0]:.2f} \\times Area) + ({coefficients[1]:.2f} \\times Rooms) + ({coefficients[2]:.2f} \\times Kitchens)")

# User inputs
st.sidebar.header('Input Features')
input_area = st.sidebar.number_input('Area (sq ft)', min_value=1000, max_value=7500, value=3000, step=100)
input_rooms = st.sidebar.slider('Number of Rooms', 2, 10, 5)
input_kitchens = st.sidebar.slider('Number of Kitchens', 1, 4, 2)

# Prediction
predicted_price = predict_price(input_area, input_rooms, input_kitchens)
st.sidebar.header('Predicted House Price')
st.sidebar.write(f'${predicted_price:.2f}')

# Enhanced Visualization
fig = px.scatter_3d(data, x='Area', y='Rooms', z='Kitchens', color='Price')
new_point = pd.DataFrame([[input_area, input_rooms, input_kitchens, predicted_price]], columns=['Area', 'Rooms', 'Kitchens', 'Price'])
fig.add_trace(go.Scatter3d(x=new_point['Area'], y=new_point['Rooms'], z=new_point['Kitchens'],
                           mode='markers', marker=dict(size=10, color='red'),
                           name='Input Prediction'))

fig.update_layout(title='House Pricing Model Visualization',
                  scene=dict(
                      xaxis_title='Area (sq ft)',
                      yaxis_title='Rooms',
                      zaxis_title='Kitchens'
                  ))

st.plotly_chart(fig, use_container_width=True)
