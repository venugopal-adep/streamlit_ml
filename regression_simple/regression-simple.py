import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# [Keep the page configuration and custom CSS as before]

# Title and introduction
st.title("🏋️‍♂️ Height-Weight Predictor: Linear Regression in Action")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Height-Weight Predictor! Explore the relationship between height and weight
using linear regression. Interact with the model, make predictions, and understand key concepts
in this data-driven adventure!
""")

# Functions
@st.cache_data
def generate_data(num_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    height = np.linspace(60, 75, num_samples)  # Heights ranging from 60 to 75 inches
    weight = -266.53 + 6.1376 * height + np.random.normal(0, 5, num_samples)  # Equation with added noise
    return pd.DataFrame({'Height': height, 'Weight': weight})

def train_model(data):
    model = LinearRegression()
    model.fit(data[['Height']], data['Weight'])
    return model

def plot_regression_line(data, model, prediction_point=None):
    fig = px.scatter(data, x='Height', y='Weight', title='Height vs. Weight Regression Analysis',
                     labels={'Height': 'Height (inches)', 'Weight': 'Weight (lbs)'}, opacity=0.7)
    fig.add_scatter(x=data['Height'], y=model.predict(data[['Height']]), mode='lines', name='Regression Line', line=dict(color='blue'))
    if prediction_point:
        fig.add_trace(px.scatter(x=[prediction_point[0]], y=[prediction_point[1]], labels={'x': 'Predicted Height', 'y': 'Predicted Weight'}).data[0])
        fig.data[-1].update(marker=dict(color='red', size=10), name='Prediction Point')
    fig.update_layout(template="plotly_white")
    return fig

# Initialize session state
if 'data' not in st.session_state or 'model' not in st.session_state:
    st.session_state['data'] = generate_data(100)
    st.session_state['model'] = train_model(st.session_state['data'])

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "🔬 Explore", "🎯 Predict", "📚 Learn"])

with tab1:
    st.header("Linear Regression Model")
    model = st.session_state['model']
    st.latex(f"\\hat{{Y}} = {model.intercept_:.2f} + {model.coef_[0]:.4f}X")
    st.write("Where X is the height in inches and Ŷ is the predicted weight in lbs.")
    
    predictions = model.predict(st.session_state['data'][['Height']])
    mae = mean_absolute_error(st.session_state['data']['Weight'], predictions)
    mse = mean_squared_error(st.session_state['data']['Weight'], predictions)
    r2 = r2_score(st.session_state['data']['Weight'], predictions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

with tab2:
    st.header("🔍 Data Explorer")
    
    # Move the "Generate Data" button to this tab
    if st.button("🔄 Generate New Data"):
        new_seed = np.random.randint(10000)
        st.session_state['data'] = generate_data(100, seed=new_seed)
        st.session_state['model'] = train_model(st.session_state['data'])
        st.success("New data generated and model retrained!")
    
    fig = plot_regression_line(st.session_state['data'], model)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Raw Data"):
        st.write(st.session_state['data'])

with tab3:
    st.header("🎯 Weight Predictor")
    input_height = st.slider('Height (inches)', int(st.session_state['data']['Height'].min()), int(st.session_state['data']['Height'].max()), 68)
    predicted_weight = model.predict([[input_height]])[0]
    
    st.write(f"### Predicted Weight: {predicted_weight:.2f} lbs")
    
    fig = plot_regression_line(st.session_state['data'], model, [input_height, predicted_weight])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Understanding the Intercept")
    st.write("""
    - The intercept in this regression equation is the theoretical weight when height is zero, which is not practical.
    - Extrapolating beyond the observed range of data can lead to unrealistic predictions.
    - Always consider the range of your data before applying the model to make predictions.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What does the coefficient represent in this linear regression model?",
            "options": ["The average weight", "The change in weight for each inch increase in height", "The starting weight"],
            "answer": 1,
            "explanation": "The coefficient (slope) represents how much the weight changes for each inch increase in height. In our model, it suggests that for every inch increase in height, the weight increases by about 6.14 pounds on average."
        },
        {
            "question": "Why is the intercept in this model not practically meaningful?",
            "options": ["It's always meaningful", "It represents weight at zero height, which is impossible", "It's randomly generated"],
            "answer": 1,
            "explanation": "The intercept represents the predicted weight when height is zero, which is not possible in reality. For human height-weight relationships, the intercept often doesn't have a practical interpretation because we never observe people with zero height."
        },
        {
            "question": "What does a high R² score indicate?",
            "options": ["The model is complex", "The model fits the data poorly", "The model explains a large portion of the variability in the data"],
            "answer": 2,
            "explanation": "A high R² score (close to 1) indicates that the model explains a large portion of the variability in the data. It suggests that the linear relationship between height and weight in our model accounts for a significant amount of the observed variation in weights."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(q["question"])
        user_answer = st.radio(f"Select your answer for question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['answer']:
                st.success("Correct! 🎉")
            else:
                st.error(f"Not quite. The correct answer is: {q['options'][q['answer']]}")
            
            st.markdown("**Explanation:**")
            st.write(q['explanation'])
            st.markdown("---")

st.markdown("""
## 🎓 Conclusion

Congratulations on exploring the Height-Weight Predictor! Remember:

- 📏 Linear regression helps us understand relationships between variables like height and weight.
- 🧮 The model provides insights, but always consider its limitations and the context of your data.
- 🚀 Keep exploring, keep learning, and may your predictions always be accurate!
""")
