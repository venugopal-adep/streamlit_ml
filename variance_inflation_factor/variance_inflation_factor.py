import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="VIF Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6e6;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔍 Variance Inflation Factor (VIF) Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of VIF in detecting multicollinearity!")

# Helper functions
def calculate_vif(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif.sort_values("VIF", ascending=False)

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("VIF Analysis")
    
    # Calculate VIF
    vif_df = calculate_vif(st.session_state['X'])
    
    # Display VIF table
    st.subheader("Variance Inflation Factor (VIF) Table")
    st.table(vif_df)
    
    # Create interactive bar plot
    fig = go.Figure(data=[go.Bar(x=vif_df["Feature"], y=vif_df["VIF"])])
    fig.update_layout(
        title="Variance Inflation Factor (VIF) Bar Plot",
        xaxis_title="Features",
        yaxis_title="VIF",
    )
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

with tab2:
    st.header("Solved Example: Detecting Multicollinearity")
    
    # Generate example data
    X_example = pd.DataFrame({
        'X1': np.random.randn(100),
        'X2': np.random.randn(100),
        'X3': np.random.randn(100),
        'X4': np.random.randn(100)
    })
    X_example['X5'] = 0.8 * X_example['X1'] + 0.3 * X_example['X2'] + np.random.randn(100) * 0.1
    
    vif_example = calculate_vif(X_example)
    
    st.write("We'll analyze a dataset where X5 is a linear combination of X1 and X2 with some noise.")
    
    st.table(vif_example)
    
    st.write("As we can see, X5 has a high VIF, indicating strong multicollinearity with other features.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does VIF measure?",
            "options": ["Variable importance", "Variance in features", "Multicollinearity between features", "Feature selection criteria"],
            "correct": 2,
            "explanation": "VIF measures the extent of multicollinearity between features in a dataset. It quantifies how much the variance of an estimated regression coefficient increases if your predictors are correlated."
        },
        {
            "question": "What is generally considered a high VIF value?",
            "options": ["Above 1", "Above 3", "Above 5", "Above 10"],
            "correct": 3,
            "explanation": "While there's no universal rule, a VIF above 10 is often considered to indicate high multicollinearity. Some researchers use a more conservative threshold of 5."
        },
        {
            "question": "What action might you take if a feature has a very high VIF?",
            "options": ["Always keep it in the model", "Always remove it from the model", "Consider removing it or combining it with other features", "Multiply its coefficients by the VIF"],
            "correct": 2,
            "explanation": "If a feature has a very high VIF, you might consider removing it from the model or combining it with other correlated features. However, the decision should also consider domain knowledge and the feature's importance to the problem."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
            st.write(f"Explanation: {q['explanation']}")
        st.write("---")

with tab4:
    st.header("Learn More About VIF")
    st.markdown("""
    The Variance Inflation Factor (VIF) is a measure of the amount of multicollinearity in a set of multiple regression variables. It provides an index that measures how much the variance of an estimated regression coefficient is increased because of collinearity.

    Key concepts of VIF:
    1. **Formula**: VIF for the i-th variable is 1 / (1 - R^2), where R^2 is from a regression of the i-th variable on all other predictors.
    2. **Interpretation**: A VIF of 1 means no correlation between the i-th variable and the remaining variables. A VIF between 1 and 5 suggests moderate correlation, while a VIF above 5 (or 10, depending on the source) indicates high correlation.
    3. **Usage**: VIF is commonly used in regression analysis to detect multicollinearity among predictors.

    Improved VIF strategy:
    1. **Standardization**: Always standardize your features before calculating VIF. This ensures that VIF is not affected by the scale of the variables.
    2. **Iterative Approach**: If high VIF is detected, consider removing the variable with the highest VIF and recalculating VIF for the remaining variables.
    3. **Domain Knowledge**: Always consider the importance of a variable in your model, even if it has a high VIF. Sometimes, it might be better to keep a theoretically important variable despite high VIF.
    4. **Alternative Techniques**: Consider using techniques like Principal Component Analysis (PCA) or Partial Least Squares (PLS) regression if multicollinearity is a persistent issue.

    Remember, while VIF is a useful tool for detecting multicollinearity, it should be used in conjunction with other diagnostic measures and domain expertise when making decisions about your regression model.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the use of Variance Inflation Factor (VIF) in detecting multicollinearity. Adjust the number of features, regenerate data, and explore the different tabs to learn more!")
