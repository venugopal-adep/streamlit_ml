import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Height-Weight Predictor", layout="wide", initial_sidebar_state="expanded")

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
    background-color: #e6e6fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏋️‍♂️ Height-Weight Predictor: Linear Regression in Action")
st.markdown("**Developed by: Your Name**")
st.markdown("Explore the relationship between height and weight using linear regression!")

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
        fig.add_trace(go.Scatter(x=[prediction_point[0]], y=[prediction_point[1]], mode='markers', name='Prediction Point',
                                 marker=dict(color='red', size=10)))
    fig.update_layout(template="plotly_white")
    return fig

# Initialize session state
if 'data' not in st.session_state or 'model' not in st.session_state:
    st.session_state['data'] = generate_data(100)
    st.session_state['model'] = train_model(st.session_state['data'])

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Model", "🔬 Explore", "🎯 Predict"])

with tab1:
    st.header("📚 Learning Center")
    
    st.markdown("""
    <div class="highlight">
    <h3>What is Linear Regression?</h3>
    <p>Linear regression is a statistical method used to model the relationship between two variables by fitting a linear equation to the observed data. In this case, we're using it to understand how height relates to weight.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Concepts</h3>
    <ul>
        <li><strong>Slope:</strong> Represents the change in weight for each inch increase in height</li>
        <li><strong>Intercept:</strong> The theoretical weight when height is zero (not always meaningful in practice)</li>
        <li><strong>R² Score:</strong> Measures how well the model fits the data (0 to 1, higher is better)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Limitations</h3>
    <ul>
        <li>Assumes a linear relationship, which may not always be true</li>
        <li>Sensitive to outliers</li>
        <li>May not capture complex relationships between variables</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Quiz")
    questions = [
        {
            "question": "What does the slope represent in our height-weight model?",
            "options": ["The average weight", "How much weight changes for each inch of height", "The starting weight"],
            "correct": 1,
            "explanation": "The slope represents how much the weight changes for each inch increase in height."
        },
        {
            "question": "What does a high R² score mean?",
            "options": ["The model is very complex", "The model doesn't fit the data well", "The model explains a lot of the variability in the data"],
            "correct": 2,
            "explanation": "A high R² score indicates that the model explains a large portion of the variability in the data."
        },
        {
            "question": "Why might the intercept not be meaningful in a height-weight model?",
            "options": ["It's always meaningful", "It represents weight at zero height, which is impossible", "It's randomly generated"],
            "correct": 1,
            "explanation": "The intercept represents the weight at zero height, which isn't possible for humans and thus may not have a practical interpretation."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Not quite right. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

with tab2:
    st.header("📊 Linear Regression Model")
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
    
    fig = plot_regression_line(st.session_state['data'], model)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔬 Data Explorer")
    
    if st.button("🔄 Generate New Data"):
        new_seed = np.random.randint(10000)
        st.session_state['data'] = generate_data(100, seed=new_seed)
        st.session_state['model'] = train_model(st.session_state['data'])
        st.success("New data generated and model retrained!")
    
    fig = plot_regression_line(st.session_state['data'], model)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Raw Data"):
        st.write(st.session_state['data'])
    
    st.subheader("Data Distribution")
    feature = st.selectbox("Select a feature to visualize:", ['Height', 'Weight'])
    fig_dist = px.histogram(st.session_state['data'], x=feature, nbins=20, marginal="box")
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    st.header("🎯 Weight Predictor")
    input_height = st.slider('Height (inches)', int(st.session_state['data']['Height'].min()), int(st.session_state['data']['Height'].max()), 68)
    predicted_weight = model.predict([[input_height]])[0]
    
    st.write(f"### Predicted Weight: {predicted_weight:.2f} lbs")
    
    fig = plot_regression_line(st.session_state['data'], model, [input_height, predicted_weight])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Interpretation</h3>
    <p>The red point on the graph shows your prediction. Notice how it falls on the blue regression line. 
    This line represents our model's best guess for the relationship between height and weight based on the data.</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the use of linear regression to predict weight based on height. Explore the different tabs to learn more about the model and make predictions!")
