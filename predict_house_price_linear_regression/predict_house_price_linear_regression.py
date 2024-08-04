import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="House Price Predictor: Multiple Regression Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #ffd700;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Predictor: Multiple Regression Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore how multiple features affect house prices using linear regression!")

# Helper functions
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

# Sidebar
st.sidebar.header("Data Generation")
if st.sidebar.button("🔄 Generate New Data"):
    st.session_state.data = generate_data(300)
    st.session_state.model = train_model(st.session_state.data)
    st.sidebar.success("New data generated and model retrained!")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_data(300)
    st.session_state.model = train_model(st.session_state.data)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🔬 Explore", "📊 Model", "🎯 Predict", "🧠 Quiz"])

with tab1:
    st.header("Understanding Multiple Linear Regression")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Multiple Linear Regression?</h3>
    <p>Multiple linear regression is a statistical method that uses several explanatory variables to predict the outcome of a response variable. The goal is to model the linear relationship between the explanatory (independent) variables and the response (dependent) variable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts</h3>
    <ul>
        <li><strong>Multiple Predictors:</strong> Unlike simple linear regression, multiple regression can handle several independent variables.</li>
        <li><strong>Coefficients:</strong> Each predictor has its own coefficient, representing its individual effect on the outcome.</li>
        <li><strong>Interpretation:</strong> The coefficient for each variable represents the change in the outcome for a one-unit change in that variable, holding all other variables constant.</li>
        <li><strong>Model Complexity:</strong> While more predictors can lead to better fit, it also increases the risk of overfitting.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Multiple Linear Regression?</h3>
    <ul>
        <li><span class="highlight">Improved Predictions:</span> By considering multiple factors, we can often make more accurate predictions.</li>
        <li><span class="highlight">Understanding Relationships:</span> It helps us understand how different factors interact to influence the outcome.</li>
        <li><span class="highlight">Real-world Applicability:</span> Many real-world phenomena are influenced by multiple factors, making this a versatile technique.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("🔬 Data Explorer")
    
    st.subheader("Sample Data")
    st.write(st.session_state.data.head())
    
    st.subheader("Data Statistics")
    st.write(st.session_state.data.describe())
    
    st.subheader("Correlation Matrix")
    fig_corr = px.imshow(st.session_state.data.corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
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

with tab4:
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

with tab5:
    st.header("🧠 Test Your Knowledge")

    questions = [
        {
            "question": "What does multiple linear regression allow us to do?",
            "options": ["Predict based on one variable", "Predict based on multiple variables", "Only work with categorical data"],
            "correct": 1,
            "explanation": "Multiple linear regression allows us to predict an outcome based on multiple variables, which can lead to more accurate predictions in complex scenarios."
        },
        {
            "question": "In our house price model, what does a coefficient represent?",
            "options": ["The total price of the house", "The change in price for a one-unit change in a feature", "The number of features"],
            "correct": 1,
            "explanation": "A coefficient in our model represents the change in house price for a one-unit change in the corresponding feature, holding all other features constant."
        },
        {
            "question": "Why might adding more features to a regression model not always improve it?",
            "options": ["It always improves the model", "It can lead to overfitting", "It makes the model slower"],
            "correct": 1,
            "explanation": "While adding more features can improve a model's fit to the training data, it can also lead to overfitting, where the model performs poorly on new, unseen data."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Great job!")
            else:
                st.error("Not quite. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates multiple linear regression for house price prediction. Generate new data, explore the model, and make predictions!")
