import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import probplot

# Set page config
st.set_page_config(page_title="Linear Regression Assumptions Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🔬 Linear Regression Assumptions Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the key assumptions of linear regression and their importance!")

# Helper functions
@st.cache_data
def generate_data(n, noise, non_linearity=False, multicollinearity_factor=0):
    np.random.seed(42)
    X1 = np.random.normal(0, 1, n)
    X2 = X1 * multicollinearity_factor + np.random.normal(0, 1, n)
    if non_linearity:
        Y = np.square(X1) + 3 * X2 + np.random.normal(0, noise, n)
    else:
        Y = 2 * X1 + 3 * X2 + np.random.normal(0, noise, n)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

def plot_linearity_test(data):
    fig = px.scatter(data, x='X1', y='Y', trendline='ols', title="Linearity Test: X1 vs Y")
    fig.update_layout(template="plotly_white")
    return fig

def plot_multicollinearity_test(data):
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    fig.update_layout(template="plotly_white")
    return fig

def plot_heteroscedasticity_test(data):
    model = LinearRegression()
    model.fit(data[['X1', 'X2']], data['Y'])
    predictions = model.predict(data[['X1', 'X2']])
    residuals = data['Y'] - predictions
    fig = px.scatter(x=predictions, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residuals vs Predicted")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(template="plotly_white")
    return fig

def plot_normality_test(residuals):
    qq = probplot(residuals, dist="norm")
    theoretical_quantiles = qq[0][0]
    ordered_values = qq[0][1]
    slope, intercept = qq[1][0], qq[1][1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles * slope + intercept, mode='lines', name='Fit Line'))
    fig.update_layout(title="Q-Q Plot of Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Ordered Values",
                      showlegend=True, template="plotly_white")
    return fig

# Sidebar
st.sidebar.header("Data Configuration")
n = st.sidebar.slider("Number of Observations", 100, 1000, 500)
noise = st.sidebar.slider("Noise Level", 0.1, 10.0, 1.0, 0.1)
non_linearity = st.sidebar.checkbox("Introduce Non-linearity")
multicollinearity_factor = st.sidebar.slider("Multicollinearity Factor", 0.0, 1.0, 0.0, 0.1)

# Generate data
if 'data' not in st.session_state or st.sidebar.button("Generate New Data"):
    st.session_state.data = generate_data(n, noise, non_linearity, multicollinearity_factor)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data", "📈 Assumptions", "🧮 Analysis", "🧠 Quiz"])

with tab1:
    st.header("Understanding Linear Regression Assumptions")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What are Linear Regression Assumptions?</h3>
    <p>Linear regression is based on several key assumptions that ensure the model's reliability and effectiveness:</p>
    <ul>
        <li><strong>Linearity:</strong> The relationship between variables is linear.</li>
        <li><strong>Independence:</strong> Observations are independent of each other.</li>
        <li><strong>Homoscedasticity:</strong> The variance of residuals is constant across all levels of the independent variables.</li>
        <li><strong>Normality:</strong> The residuals are normally distributed.</li>
        <li><strong>No Multicollinearity:</strong> The independent variables are not highly correlated with each other.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why are these Assumptions Important?</h3>
    <ul>
        <li><span class="highlight">Reliable Predictions:</span> Ensure that the model's predictions are accurate and trustworthy.</li>
        <li><span class="highlight">Valid Inference:</span> Allow for valid statistical inference about the population.</li>
        <li><span class="highlight">Model Performance:</span> Help in building a model that performs well on unseen data.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Generated Data")
        st.write(st.session_state.data.head())
    with col2:
        st.subheader("Data Summary")
        st.write(st.session_state.data.describe())

with tab3:
    st.header("📈 Regression Assumptions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linearity Test")
        st.plotly_chart(plot_linearity_test(st.session_state.data), use_container_width=True)
        st.write("A clear linear trend suggests the linearity assumption is met.")
    
    with col2:
        st.subheader("Heteroscedasticity Test")
        st.plotly_chart(plot_heteroscedasticity_test(st.session_state.data), use_container_width=True)
        st.write("Random scatter suggests homoscedasticity (constant variance).")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Multicollinearity Test")
        st.plotly_chart(plot_multicollinearity_test(st.session_state.data[['X1', 'X2']]), use_container_width=True)
        st.write("Low correlation between independent variables is desirable.")
    
    with col4:
        st.subheader("Normality of Residuals Test")
        residuals = st.session_state.data['Y'] - LinearRegression().fit(st.session_state.data[['X1', 'X2']], st.session_state.data['Y']).predict(st.session_state.data[['X1', 'X2']])
        st.plotly_chart(plot_normality_test(residuals), use_container_width=True)
        st.write("Points following the diagonal line suggest normally distributed residuals.")

with tab4:
    st.header("🧮 Detailed Analysis")
    
    st.subheader("Variance Inflation Factor (VIF)")
    vif_data = calculate_vif(st.session_state.data[['X1', 'X2']])
    st.write(vif_data)
    st.write("VIF > 5 indicates high multicollinearity")
    
    st.subheader("Correlation Matrix")
    corr_matrix = st.session_state.data.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does the linearity assumption in linear regression mean?",
            "options": ["The data points form a straight line", "There's a linear relationship between variables", "The residuals are linear", "All of the above"],
            "correct": 1,
            "explanation": "The linearity assumption means there's a linear relationship between the independent variables and the dependent variable."
        },
        {
            "question": "What does homoscedasticity mean?",
            "options": ["The residuals are normally distributed", "The variance of residuals is constant", "The independent variables are not correlated", "The relationship is linear"],
            "correct": 1,
            "explanation": "Homoscedasticity means the variance of residuals is constant across all levels of the independent variables."
        },
        {
            "question": "What's a sign of multicollinearity in the data?",
            "options": ["High VIF values", "Non-linear relationships", "Heteroscedasticity", "Non-normal residuals"],
            "correct": 0,
            "explanation": "High Variance Inflation Factor (VIF) values indicate multicollinearity, which means the independent variables are highly correlated with each other."
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
st.sidebar.info("This app demonstrates linear regression assumptions. Adjust the settings, generate new data, and explore the different tabs to learn more!")
