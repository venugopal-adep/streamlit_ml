import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import f

def generate_data(num_samples, homoscedastic=True):
    X = np.random.rand(num_samples)
    if homoscedastic:
        noise = np.random.normal(0, 0.1, num_samples)
    else:
        noise = np.random.normal(0, X, num_samples)
    y = 2 * X + 1 + noise
    return X, y

def goldfeld_quandt_test(X, y):
    n = len(X)
    k = int(n / 3)
    X_sorted = np.sort(X)
    X_low, X_high = X_sorted[:k], X_sorted[-k:]
    y_low = y[np.argsort(X)[:k]]
    y_high = y[np.argsort(X)[-k:]]
    
    ssr_low = np.sum((y_low - np.mean(y_low))**2)
    ssr_high = np.sum((y_high - np.mean(y_high))**2)
    
    f_value = ssr_high / ssr_low
    p_value = f.sf(f_value, k-1, k-1)
    
    return f_value, p_value

def plot_data(X, y, regression_line, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=X, y=regression_line, mode='lines', name='Regression Line'))
    fig.update_layout(title=title, xaxis_title='X', yaxis_title='y')
    st.plotly_chart(fig)

def main():
    st.title("Homoscedasticity Demonstration")
    
    st.sidebar.header("Parameters")
    num_samples = st.sidebar.slider("Number of Samples", 50, 500, 100, 50)
    data_type = st.sidebar.radio("Data Type", ("Homoscedastic", "Heteroscedastic"))
    
    homoscedastic = data_type == "Homoscedastic"
    X, y = generate_data(num_samples, homoscedastic)
    
    regression_line = 2 * X + 1
    
    plot_data(X, y, regression_line, data_type + " Data")
    
    f_value, p_value = goldfeld_quandt_test(X, y)
    
    st.subheader("Goldfeld-Quandt Test Results")
    st.write(f"F-value: {f_value:.2f}")
    st.write(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. The residuals are heteroscedastic.")
    else:
        st.write("Fail to reject the null hypothesis. The residuals are homoscedastic.")

if __name__ == "__main__":
    main()
