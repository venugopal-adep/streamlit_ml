import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import probplot

# Set page configuration
st.set_page_config(page_title="Linear Regression Assumptions", layout="wide")

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
st.title("🔬 Linear Regression Assumptions")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
Welcome to the Statistical Analysis Tool! This interactive application helps you explore 
fundamental statistical concepts through visualizations. Adjust the parameters in the sidebar
to see how they affect various statistical tests and assumptions.
""")

# Functions
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

# Sidebar controls
st.sidebar.header("Data Options")
n = st.sidebar.slider("Number of Observations", 100, 1000, 500)
noise = st.sidebar.slider("Noise Level", 0.1, 10.0, 1.0, 0.1)
non_linearity = st.sidebar.checkbox("Introduce Non-linearity")
multicollinearity_factor = st.sidebar.slider("Multicollinearity Factor", 0.0, 1.0, 0.0, 0.1)

# Generate data based on sidebar inputs
data = generate_data(n, noise, non_linearity, multicollinearity_factor)

# Main content using tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "📈 Linearity", "🔗 Multicollinearity", "🔄 Heteroscedasticity", "🔔 Normality"])

with tab1:
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Data")
        st.write(data.head())
        
    with col2:
        st.subheader("Data Summary")
        st.write(data.describe())

with tab2:
    st.header("📈 Linearity Test")
    
    st.write("""
    The linearity test checks if there's a linear relationship between the independent variables (X) and the dependent variable (Y).
    A clear linear trend in the scatter plot suggests a linear relationship.
    """)
    
    st.plotly_chart(plot_linearity_test(data), use_container_width=True)

with tab3:
    st.header("🔗 Multicollinearity Test")
    
    st.write("""
    Multicollinearity occurs when independent variables are highly correlated with each other.
    We use two methods to detect multicollinearity:
    1. Correlation Heatmap
    2. Variance Inflation Factor (VIF)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        st.plotly_chart(plot_multicollinearity_test(data[['X1', 'X2']]), use_container_width=True)
    
    with col2:
        st.subheader("Variance Inflation Factor (VIF)")
        vif_data = calculate_vif(data[['X1', 'X2']])
        st.write(vif_data)
        st.write("VIF > 5 indicates high multicollinearity")

with tab4:
    st.header("🔄 Heteroscedasticity Test")
    
    st.write("""
    Heteroscedasticity occurs when the variability of a variable is unequal across the range of values of a second variable that predicts it.
    In the plot below, a random scatter of points indicates homoscedasticity (desired), while any pattern suggests heteroscedasticity.
    """)
    
    st.plotly_chart(plot_heteroscedasticity_test(data), use_container_width=True)

with tab5:
    st.header("🔔 Normality of Residuals Test")
    
    st.write("""
    This test checks if the residuals (differences between observed and predicted values) are normally distributed.
    In a Q-Q plot, points following the diagonal line suggest normally distributed residuals.
    """)
    
    residuals = data['Y'] - LinearRegression().fit(data[['X1', 'X2']], data['Y']).predict(data[['X1', 'X2']])
    st.plotly_chart(plot_normality_test(residuals), use_container_width=True)

st.markdown("""
## 🎓 Conclusion

Congratulations on exploring various statistical tests and assumptions! Remember:

- 📊 Always check these assumptions when performing regression analysis.
- 🔍 Violations of these assumptions can lead to unreliable results.
- 📈 Adjusting your model or data might be necessary if assumptions are violated.

Keep exploring and refining your understanding of statistical analysis!
""")
