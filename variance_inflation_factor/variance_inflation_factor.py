import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="VIF Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🔍 Variance Inflation Factor (VIF) Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of VIF in detecting multicollinearity!")

# Helper functions
@st.cache_data
def calculate_vif(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif.sort_values("VIF", ascending=False)

@st.cache_data
def generate_data(num_features, num_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(num_samples, num_features)
    for i in range(1, num_features):
        if np.random.rand() > 0.5:
            prev_feature = np.random.randint(0, i)
            weight = np.random.rand() * 0.9 + 0.1
            noise = np.random.rand() * 0.1
            X[:, i] = X[:, prev_feature] * weight + np.random.normal(0, noise, num_samples)
    columns = [f"Feature {i+1}" for i in range(num_features)]
    return pd.DataFrame(X, columns=columns)

# Sidebar
st.sidebar.header("Configuration")
num_features = st.sidebar.slider("Number of Features", min_value=4, max_value=10, value=5)

# Generate data
if 'X' not in st.session_state or st.sidebar.button('Regenerate Data'):
    new_seed = np.random.randint(10000)
    st.session_state['X'] = generate_data(num_features, seed=new_seed)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualization", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About VIF")
    
    st.markdown("""
    <div class="highlight">
    <h3>What is Variance Inflation Factor (VIF)?</h3>
    <p>VIF is a measure of the amount of multicollinearity in a set of multiple regression variables. It tells us how much the variance of an estimated regression coefficient increases if your predictors are correlated.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>How to Interpret VIF?</h3>
    <ul>
        <li>VIF = 1: No correlation between the feature and other features</li>
        <li>1 < VIF < 5: Moderate correlation</li>
        <li>VIF > 5: High correlation (some use a threshold of 10)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why is VIF Important?</h3>
    <ul>
        <li>Helps detect multicollinearity in regression analysis</li>
        <li>Multicollinearity can lead to unstable and unreliable regression estimates</li>
        <li>Guides feature selection and model improvement</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 VIF Analysis")
    
    # Calculate VIF
    vif_df = calculate_vif(st.session_state['X'])
    
    # Display VIF table
    st.subheader("Variance Inflation Factor (VIF) Table")
    st.table(vif_df)
    
    # Create interactive bar plot
    fig = px.bar(vif_df, x="Feature", y="VIF", title="Variance Inflation Factor (VIF) Bar Plot")
    fig.update_layout(xaxis_title="Features", yaxis_title="VIF")
    st.plotly_chart(fig, use_container_width=True)
    
    # VIF interpretation
    st.subheader("VIF Interpretation")
    for _, row in vif_df.iterrows():
        if row['VIF'] < 5:
            st.success(f"{row['Feature']}: VIF = {row['VIF']:.2f} - Low multicollinearity")
        elif row['VIF'] < 10:
            st.warning(f"{row['Feature']}: VIF = {row['VIF']:.2f} - Moderate multicollinearity")
        else:
            st.error(f"{row['Feature']}: VIF = {row['VIF']:.2f} - High multicollinearity")

    st.markdown("""
    <div class="highlight">
    <p>The bar plot shows the VIF for each feature. Longer bars indicate higher VIF values, suggesting stronger multicollinearity.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("🧮 Example: Detecting Multicollinearity")
    
    # Generate example data
    X_example = pd.DataFrame({
        'X1': np.random.randn(100),
        'X2': np.random.randn(100),
        'X3': np.random.randn(100),
        'X4': np.random.randn(100)
    })
    X_example['X5'] = 0.8 * X_example['X1'] + 0.3 * X_example['X2'] + np.random.randn(100) * 0.1
    
    vif_example = calculate_vif(X_example)
    
    st.write("Let's analyze a dataset where X5 is a linear combination of X1 and X2 with some noise.")
    
    fig = px.bar(vif_example, x="Feature", y="VIF", title="VIF for Example Dataset")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="highlight">
    <p>Notice how X5 has a much higher VIF than the other features. This indicates strong multicollinearity with other features, particularly X1 and X2.</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does VIF stand for?",
            "options": ["Very Important Factor", "Variance Inflation Factor", "Variable Influence Factor", "Value Increase Factor"],
            "correct": 1,
            "explanation": "VIF stands for Variance Inflation Factor. It measures how much the variance of a regression coefficient is inflated due to multicollinearity in the model."
        },
        {
            "question": "What does a VIF value of 1 indicate?",
            "options": ["High multicollinearity", "Moderate multicollinearity", "No multicollinearity", "Perfect multicollinearity"],
            "correct": 2,
            "explanation": "A VIF value of 1 indicates that there is no correlation between this feature and the other features. This is the ideal scenario."
        },
        {
            "question": "Why is multicollinearity a problem in regression analysis?",
            "options": ["It always improves model performance", "It can lead to unstable and unreliable estimates", "It always decreases model performance", "It has no effect on the model"],
            "correct": 1,
            "explanation": "Multicollinearity can lead to unstable and unreliable regression estimates. When predictors are highly correlated, it becomes difficult to determine the individual effect of each predictor on the response variable."
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

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the use of Variance Inflation Factor (VIF) in detecting multicollinearity. Adjust the number of features, regenerate data, and explore the different tabs to learn more!")
