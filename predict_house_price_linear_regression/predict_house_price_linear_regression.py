import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        background-color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stTab {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🏠 House Price Predictor: Multiple Regression Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the House Price Predictor! Explore how different features like area, number of rooms,
and kitchens affect house prices using multiple linear regression. Interact with the model,
make predictions, and visualize the results in 3D!
""")

# Functions
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

def train_model(data):
    model = LinearRegression()
    model.fit(data[['Area', 'Rooms', 'Kitchens']], data['Price'])
    return model

def predict_price(model, area, rooms, kitchens):
    return model.predict(np.array([[area, rooms, kitchens]]))[0]

def create_3d_plot(data, input_area, input_rooms, input_kitchens, predicted_price):
    fig = px.scatter_3d(data, x='Area', y='Rooms', z='Kitchens', color='Price')
    new_point = pd.DataFrame([[input_area, input_rooms, input_kitchens, predicted_price]], 
                             columns=['Area', 'Rooms', 'Kitchens', 'Price'])
    fig.add_trace(go.Scatter3d(x=new_point['Area'], y=new_point['Rooms'], z=new_point['Kitchens'],
                               mode='markers', marker=dict(size=10, color='red'),
                               name='Input Prediction'))
    fig.update_layout(title='House Pricing Model Visualization',
                      scene=dict(
                          xaxis_title='Area (sq ft)',
                          yaxis_title='Rooms',
                          zaxis_title='Kitchens'
                      ))
    return fig

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_data(300)
    st.session_state.model = train_model(st.session_state.data)

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Explore", "📊 Model", "🎯 Predict", "📚 Learn"])

with tab1:
    st.header("🔬 Data Explorer")
    
    if st.button("🔄 Generate New Data"):
        st.session_state.data = generate_data(300)
        st.session_state.model = train_model(st.session_state.data)
        st.success("New data generated and model retrained!")
    
    st.subheader("Sample Data")
    st.write(st.session_state.data.head())
    
    st.subheader("Data Statistics")
    st.write(st.session_state.data.describe())
    
    st.subheader("Correlation Matrix")
    fig_corr = px.imshow(st.session_state.data.corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.header("📊 Model Details")
    
    coefficients = st.session_state.model.coef_
    intercept = st.session_state.model.intercept_
    
    st.subheader("Regression Equation")
    st.latex(f"Price = {intercept:.2f} + ({coefficients[0]:.2f} \\times Area) + ({coefficients[1]:.2f} \\times Rooms) + ({coefficients[2]:.2f} \\times Kitchens)")
    
    st.subheader("Interpretation")
    st.write(f"""
    - For every 1 sq ft increase in area, the price is expected to increase by ${coefficients[0]:.2f}.
    - For each additional room, the price is expected to increase by ${coefficients[1]:.2f}.
    - For each additional kitchen, the price is expected to increase by ${coefficients[2]:.2f}.
    - When all features are 0, the expected price is ${intercept:.2f} (this is the y-intercept).
    """)

with tab3:
    st.header("🎯 Price Predictor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_area = st.number_input('Area (sq ft)', min_value=1000, max_value=7500, value=3000, step=100)
    with col2:
        input_rooms = st.slider('Number of Rooms', 2, 10, 5)
    with col3:
        input_kitchens = st.slider('Number of Kitchens', 1, 4, 2)
    
    predicted_price = predict_price(st.session_state.model, input_area, input_rooms, input_kitchens)
    
    st.subheader("Predicted House Price")
    st.write(f"${predicted_price:.2f}")
    
    fig_3d = create_3d_plot(st.session_state.data, input_area, input_rooms, input_kitchens, predicted_price)
    st.plotly_chart(fig_3d, use_container_width=True)

with tab4:
    st.header("📚 Learning Center")
    
    st.subheader("Understanding Multiple Linear Regression")
    st.write("""
    Multiple linear regression is an extension of simple linear regression used to predict an outcome variable (y) based on multiple distinct predictor variables (x). Here's what you need to know:
    
    1. **Multiple Predictors**: Unlike simple linear regression, multiple regression can handle several independent variables.
    2. **Coefficients**: Each predictor has its own coefficient, representing its individual effect on the outcome.
    3. **Interpretation**: The coefficient for each variable represents the change in the outcome for a one-unit change in that variable, holding all other variables constant.
    4. **Model Complexity**: While more predictors can lead to better fit, it also increases the risk of overfitting.
    """)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "In our model, which feature seems to have the largest impact on house price?",
            "options": ["Area", "Number of Rooms", "Number of Kitchens"],
            "answer": 1,
            "explanation": "Based on the coefficients in our model, the number of rooms has the largest impact on price. Each additional room increases the price by about $25,000, which is more significant than the impact of area or kitchens."
        },
        {
            "question": "What does the intercept represent in this context?",
            "options": ["The minimum house price", "The price of a house with no area, rooms, or kitchens", "The average house price"],
            "answer": 1,
            "explanation": "The intercept represents the theoretical price of a house with 0 area, 0 rooms, and 0 kitchens. In real-world scenarios, this often doesn't have a practical interpretation, as we never observe houses with zero of all features."
        },
        {
            "question": "Why might we use multiple regression instead of simple regression for house price prediction?",
            "options": ["It's always more accurate", "It allows us to consider multiple factors that influence price", "It's easier to interpret"],
            "answer": 1,
            "explanation": "Multiple regression allows us to consider multiple factors that influence house prices simultaneously. This can lead to more accurate predictions as real-world housing prices are influenced by many factors, not just a single variable."
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

Congratulations on exploring the House Price Predictor! Remember:

- 📊 Multiple linear regression allows us to consider various factors affecting house prices.
- 🧮 Each coefficient represents the impact of a specific feature on the price.
- 🚀 While powerful, always consider the limitations and assumptions of your model.

Keep exploring and happy predicting!
""")
