import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Statistical Learning vs Machine Learning")

st.markdown("""
The main difference between statistical learning and machine learning lies in their focus and goals:
- Statistical Learning: Focuses on understanding the relationships between variables and making inferences.
- Machine Learning: Focuses on making accurate predictions using complex models.

In this interactive demo, we'll illustrate this difference using a simple linear regression example.
""")

def generate_data():
    np.random.seed(np.random.randint(100))
    X = np.random.rand(100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1)
    return X, y

def update_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Update statistical learning model
    model_stats.fit(X_train, y_train)

    # Update machine learning model
    model_ml.fit(X_train, y_train)

    # Update plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train.squeeze(), y=y_train.squeeze(), mode='markers', name='Training Data'))

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model_ml.predict(x_range.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=x_range.squeeze(), y=y_range.squeeze(), mode='lines', name='Regression Line'))

    fig.update_layout(title='Machine Learning Prediction',
                      xaxis_title='X',
                      yaxis_title='y')

    st.plotly_chart(fig)

    # Update coefficients and intercept
    st.write("Statistical Model Coefficients:")
    st.write(f"Intercept: {model_stats.intercept_[0]:.2f}")
    st.write(f"Coefficient: {model_stats.coef_[0][0]:.2f}")

# Create models
model_stats = LinearRegression()
model_ml = LinearRegression()

# Generate initial data
X, y = generate_data()
update_models(X, y)

# Sidebar button
if st.sidebar.button("Regenerate Data"):
    X, y = generate_data()
    update_models(X, y)

st.subheader("Statistical Learning")
st.markdown("""
In statistical learning, we focus on understanding the relationships between variables. The coefficients of the linear regression model provide insights into how the input variable (X) is related to the output variable (y).
""")

st.subheader("Machine Learning")
st.markdown("""
In machine learning, the focus is on making accurate predictions. The trained model is used to predict the value of y for new input values of X. The regression line is plotted to visualize the model's predictions.
""")

st.header("Conclusion")
st.markdown("""
In summary:
- Statistical learning emphasizes understanding the relationships between variables and making inferences based on the model's coefficients.
- Machine learning focuses on building models that can make accurate predictions on new, unseen data points.

Both approaches have their own merits and are used in different scenarios depending on the goals of the analysis.
""")