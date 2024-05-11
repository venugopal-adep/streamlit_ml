import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define four datasets
data1 = pd.DataFrame({
    'Height': [63, 64, 66, 69, 69, 71, 71, 72, 73, 75],
    'Weight': [127, 121, 142, 157, 162, 156, 169, 165, 181, 208]
})
data2 = pd.DataFrame({
    'Height': [62, 65, 67, 70, 72, 74, 76, 78, 80, 82],
    'Weight': [130, 135, 145, 160, 175, 190, 205, 220, 235, 250]
})
data3 = pd.DataFrame({
    'Height': [60, 62, 63, 66, 68, 69, 72, 74, 76, 78],
    'Weight': [110, 115, 120, 130, 140, 150, 160, 170, 180, 190]
})
data4 = pd.DataFrame({
    'Height': [58, 59, 61, 63, 65, 67, 67, 71, 73, 77],
    'Weight': [95, 100, 102, 112, 123, 135, 145, 157, 165, 172]
})

# Function to toggle between datasets
def load_data(choice):
    if choice == 'Dataset 1':
        return data1
    elif choice == 'Dataset 2':
        return data2
    elif choice == 'Dataset 3':
        return data3
    elif choice == 'Dataset 4':
        return data4

# Sidebar for dataset selection
st.sidebar.header('Choose a Dataset')
dataset_choice = st.sidebar.radio("Select Dataset", ('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))
data = load_data(dataset_choice)

# Train the regression model
model = LinearRegression()
model.fit(data[['Height']], data['Weight'])

# Prediction and residuals
data['Predicted Weight'] = model.predict(data[['Height']])
data['Residual'] = data['Weight'] - data['Predicted Weight']
data['Residual^2'] = data['Residual'] ** 2

# Calculate metrics
mae = mean_absolute_error(data['Weight'], data['Predicted Weight'])
mse = mean_squared_error(data['Weight'], data['Predicted Weight'])
r2 = r2_score(data['Weight'], data['Predicted Weight'])

# Visualization function for data and residuals
def plot_data_with_residuals(data):
    fig = px.scatter(data, x='Height', y='Weight', title='Height vs. Weight Regression Analysis',
                     labels={'Height': 'Height (inches)', 'Weight': 'Weight (lbs)'}, opacity=0.7)
    fig.add_scatter(x=data['Height'], y=data['Predicted Weight'], mode='lines', name='Regression Line')
    for i, row in data.iterrows():
        fig.add_shape(type='line', x0=row['Height'], y0=row['Weight'], x1=row['Height'], y1=row['Predicted Weight'],
                      line=dict(color='red', width=1))
    return fig

# Display Regression Equation and Metrics
st.write("## Linear Regression : Residual Error")
st.write("**Developed by : Venugopal Adep**")
st.latex(f"\\hat{{Y}} = {model.intercept_:.2f} + {model.coef_[0]:.4f}X")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot
fig = plot_data_with_residuals(data)
st.plotly_chart(fig)

# Display data table with predictions and residuals
st.write("### Data Table with Predictions and Residuals")
st.write(data)

# Explanation of regression and residuals
st.write("""
### Understanding the Model
- The regression model is obtained by minimizing the sum of squared residuals.
- The sum of residuals is always close to zero, indicating a balanced distribution around the regression line.
- Each line in the plot from a point to the regression line represents a residual.
""")
