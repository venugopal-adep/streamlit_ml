import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def generate_data(num_features, num_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Generate initial random features
    X = np.random.randn(num_samples, num_features)
    
    # Introduce varying degrees of multicollinearity randomly
    for i in range(1, num_features):
        if np.random.rand() > 0.5:  # Randomly decide whether to introduce multicollinearity
            # Create a linear combination of this feature with a previous one, with random weights
            prev_feature = np.random.randint(0, i)
            weight = np.random.rand() * 0.9 + 0.1  # Ensure some weight between 0.1 and 1.0
            noise = np.random.rand() * 0.1  # Small noise
            X[:, i] = X[:, prev_feature] * weight + np.random.normal(0, noise, num_samples)
    
    columns = [f"Feature {i+1}" for i in range(num_features)]
    return pd.DataFrame(X, columns=columns)

def main():
    st.write("## Variance Inflation Factor (VIF) Demo")
    st.write("**Developed by : Venugopal Adep**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    num_features = st.sidebar.slider("Number of Features", min_value=4, max_value=10, value=5)
    
    # Button to regenerate data
    if st.sidebar.button('Regenerate Data'):
        new_seed = np.random.randint(10000)
        st.session_state['X'] = generate_data(num_features, seed=new_seed)

    # Initialize data if not in session state
    if 'X' not in st.session_state:
        st.session_state['X'] = generate_data(num_features)

    # Calculate VIF
    vif_df = calculate_vif(st.session_state['X'])
    
    # Display VIF table
    st.write("##### Variance Inflation Factor (VIF) Table")
    st.table(vif_df)
    
    # Create interactive bar plot
    fig = go.Figure(data=[go.Bar(x=vif_df["Feature"], y=vif_df["VIF"])])
    fig.update_layout(
        title="Variance Inflation Factor (VIF) Bar Plot",
        xaxis_title="Features",
        yaxis_title="VIF",
    )
    st.plotly_chart(fig)
    
    # Explanation of VIF
    st.subheader("What is VIF?")
    st.write("""
    The Variance Inflation Factor (VIF) quantifies the extent of correlation between one predictor and the other predictors in a model. It is used to diagnose multicollinearity in regression analyses.
    """)
    st.write("### VIF Formula:")
    st.latex(r"\text{VIF}_i = \frac{1}{1 - R_i^2}")
    st.write("""
    Here, \( R_i^2 \) is the R-squared value obtained by regressing the \( i^{th} \) predictor on all other predictors.
    """)
    
    # Interpretation guidelines
    st.subheader("Interpretation Guidelines")
    st.markdown("""
    - VIF = 1: No multicollinearity
    - VIF > 1 and VIF < 5: Moderate multicollinearity
    - VIF >= 5: High multicollinearity (may need to remove or combine features)
    """)

if __name__ == "__main__":
    main()
