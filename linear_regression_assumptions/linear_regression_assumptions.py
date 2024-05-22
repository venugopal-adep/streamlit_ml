import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from scipy.stats import probplot
from scipy import stats

# Explanation text
st.title("Statistical Analysis Tool")
st.write('**Developed by : Venugopal Adep**')
st.markdown("""
This tool helps you explore fundamental statistical concepts through interactive visualizations. 
Adjust the parameters in the sidebar to see how they affect the results.
""")

# Function to generate synthetic data
def generate_data(n, noise, non_linearity=False, multicollinearity_factor=0):
    np.random.seed(42)
    X1 = np.random.normal(0, 1, n)
    X2 = X1 * multicollinearity_factor + np.random.normal(0, 1, n)
    if non_linearity:
        Y = np.square(X1) + 3 * X2 + np.random.normal(0, noise, n)
    else:
        Y = 2 * X1 + 3 * X2 + np.random.normal(0, noise, n)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Calculating VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Plotting functions
def plot_linearity_test(data):
    fig = px.scatter(data, x='X1', y='Y', trendline='ols')
    return fig

def plot_multicollinearity_test(data):
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    return fig

def plot_heteroscedasticity_test(data):
    model = LinearRegression()
    model.fit(data[['X1', 'X2']], data['Y'])
    predictions = model.predict(data[['X1', 'X2']])
    residuals = data['Y'] - predictions
    fig = px.scatter(x=predictions, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Residuals vs Predicted")
    fig.add_hline(y=0, line_dash="dot")
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
                      showlegend=True)
    return fig

# Streamlit interface
st.sidebar.header("Data Options")
n = st.sidebar.slider("Number of Observations", 100, 1000, 500)
noise = st.sidebar.slider("Noise Level", 0.1, 10.0, 1.0, 0.1)
non_linearity = st.sidebar.checkbox("Introduce Non-linearity")
multicollinearity_factor = st.sidebar.slider("Multicollinearity Factor", 0.0, 1.0, 0.0, 0.1)

data = generate_data(n, noise, non_linearity, multicollinearity_factor)
vif_data = calculate_vif(data[['X1', 'X2']])

st.write("## Linearity Test")
st.plotly_chart(plot_linearity_test(data))

st.write("## Multicollinearity Test")
st.plotly_chart(plot_multicollinearity_test(data[['X1', 'X2']]))
st.write(vif_data)

st.write("## Heteroscedasticity Test")
st.plotly_chart(plot_heteroscedasticity_test(data))

residuals = data['Y'] - LinearRegression().fit(data[['X1', 'X2']], data['Y']).predict(data[['X1', 'X2']])
st.write("## Normality of Residuals Test")
st.plotly_chart(plot_normality_test(residuals))
