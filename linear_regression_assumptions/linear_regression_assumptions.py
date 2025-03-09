import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set page configuration
st.set_page_config(
    page_title="Linear Regression Assumptions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Assumptions of Linear Regression")
st.markdown("""
This interactive app demonstrates the four key assumptions of linear regression:
1. Linearity between dependent and independent variables
2. No multicollinearity in independent variables
3. Homoscedasticity (constant variance of residuals)
4. Normally distributed residuals
""")

# Generate sample data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 200
    
    # For linearity demonstration
    x_linear = np.random.uniform(0, 10, n_samples)
    y_linear = 2 * x_linear + 1 + np.random.normal(0, 1, n_samples)
    
    # For multicollinearity demonstration
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = x1 * 0.8 + np.random.normal(0, 1, n_samples)  # Correlated with x1
    x3 = np.random.uniform(0, 10, n_samples)  # Independent
    
    # For heteroscedasticity demonstration
    x_hetero = np.random.uniform(0, 10, n_samples)
    y_hetero = 2 * x_hetero + np.random.normal(0, 0.5 + 0.5 * x_hetero, n_samples)
    
    # For normality demonstration
    x_norm = np.random.uniform(0, 10, n_samples)
    y_norm = 2 * x_norm + 1 + np.random.normal(0, 1, n_samples)
    
    # For non-linear data
    x_nonlinear = np.random.uniform(1, 10, n_samples)
    y_nonlinear = np.exp(0.3 * x_nonlinear) + np.random.normal(0, 2, n_samples)
    
    # Create dataframes
    df_linear = pd.DataFrame({'x': x_linear, 'y': y_linear})
    df_multi = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
    df_hetero = pd.DataFrame({'x': x_hetero, 'y': y_hetero})
    df_norm = pd.DataFrame({'x': x_norm, 'y': y_norm})
    df_nonlinear = pd.DataFrame({'x': x_nonlinear, 'y': y_nonlinear})
    
    return df_linear, df_multi, df_hetero, df_norm, df_nonlinear

df_linear, df_multi, df_hetero, df_norm, df_nonlinear = generate_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
assumption = st.sidebar.radio(
    "Select an assumption to explore:",
    ["Linearity", "No Multicollinearity", "Homoscedasticity", "Normality of Residuals"]
)

# Main content based on selected assumption
if assumption == "Linearity":
    st.header("1. Linearity Assumption")
    
    st.markdown("""
    **Assumption**: There should be a linear relationship between dependent and independent variables.
    
    **How to test**: Create scatter plots of each independent variable with the dependent variable.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Linear Relationship Example")
        
        # Interactive scatter plot with regression line
        fig = px.scatter(df_linear, x='x', y='y', trendline="ols",
                         title="Scatter Plot with Regression Line")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Non-Linear Relationship Example")
        
        # Interactive scatter plot with non-linear data
        fig = px.scatter(df_nonlinear, x='x', y='y',
                         title="Non-Linear Relationship")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("How to Fix Non-Linearity")
    
    fix_option = st.selectbox(
        "Select a transformation to apply:",
        ["Log Transformation", "Square Root Transformation", "Polynomial Features"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Non-Linear Data**")
        fig = px.scatter(df_nonlinear, x='x', y='y')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        if fix_option == "Log Transformation":
            st.markdown("**Log Transformed Data**")
            df_transformed = df_nonlinear.copy()
            df_transformed['y_transformed'] = np.log(df_transformed['y'])
            fig = px.scatter(df_transformed, x='x', y='y_transformed', trendline="ols")
            fig.update_layout(height=400, title="Log(y) vs x")
            st.plotly_chart(fig, use_container_width=True)
            
        elif fix_option == "Square Root Transformation":
            st.markdown("**Square Root Transformed Data**")
            df_transformed = df_nonlinear.copy()
            df_transformed['y_transformed'] = np.sqrt(df_transformed['y'])
            fig = px.scatter(df_transformed, x='x', y='y_transformed', trendline="ols")
            fig.update_layout(height=400, title="sqrt(y) vs x")
            st.plotly_chart(fig, use_container_width=True)
            
        elif fix_option == "Polynomial Features":
            st.markdown("**Polynomial Regression**")
            df_transformed = df_nonlinear.copy()
            
            # Create polynomial features
            x = df_transformed['x'].values
            y = df_transformed['y'].values
            
            # Fit polynomial regression
            coeffs = np.polyfit(x, y, 2)
            polynomial = np.poly1d(coeffs)
            
            # Create smooth line for plotting
            x_line = np.linspace(min(x), max(x), 100)
            y_line = polynomial(x_line)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Polynomial Fit'))
            fig.update_layout(height=400, title="Polynomial Regression")
            st.plotly_chart(fig, use_container_width=True)

elif assumption == "No Multicollinearity":
    st.header("2. No Multicollinearity Assumption")
    
    st.markdown("""
    **Assumption**: Independent variables should not be highly correlated with each other.
    
    **How to test**: Create correlation heatmaps or calculate Variance Inflation Factor (VIF).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        
        # Calculate correlation matrix
        corr_matrix = df_multi.corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Variance Inflation Factor (VIF)")
        
        # Calculate VIF
        X = df_multi.copy()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Create bar chart
        fig = px.bar(vif_data, x='Variable', y='VIF', 
                     color='VIF', 
                     color_continuous_scale='Viridis',
                     title="VIF Values (>10 indicates multicollinearity)")
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=10,
            x1=2.5,
            y1=10,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("How to Fix Multicollinearity")
    
    st.markdown("""
    1. **Remove highly correlated variables**: If two variables are highly correlated, consider removing one.
    
    2. **Combine variables**: Use dimensionality reduction techniques like PCA to create new uncorrelated variables.
    
    3. **Ridge Regression**: Use regularization techniques that can handle multicollinearity.
    """)
    
    # Interactive demo for removing a correlated variable
    if st.checkbox("Show effect of removing a correlated variable"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Correlation Matrix**")
            fig = px.imshow(df_multi.corr(), text_auto=True, color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            variable_to_remove = st.selectbox("Select variable to remove:", df_multi.columns)
            df_reduced = df_multi.drop(columns=[variable_to_remove])
            
            st.markdown("**Correlation Matrix After Removal**")
            fig = px.imshow(df_reduced.corr(), text_auto=True, color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

elif assumption == "Homoscedasticity":
    st.header("3. Homoscedasticity Assumption")
    
    st.markdown("""
    **Assumption**: Residuals should have constant variance across all levels of independent variables.
    
    **How to test**: Plot residuals vs. fitted values and check for patterns.
    """)
    
    # Fit linear model
    x = df_hetero['x'].values
    y = df_hetero['y'].values
    
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Homoscedastic Example")
        
        # Create homoscedastic data
        np.random.seed(42)
        x_homo = np.random.uniform(0, 10, 200)
        y_homo = 2 * x_homo + np.random.normal(0, 1, 200)
        
        # Fit model
        slope_homo, intercept_homo = np.polyfit(x_homo, y_homo, 1)
        y_pred_homo = slope_homo * x_homo + intercept_homo
        residuals_homo = y_homo - y_pred_homo
        
        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred_homo, y=residuals_homo, mode='markers',
                                 marker=dict(color='blue', size=8, opacity=0.6),
                                 name='Residuals'))
        fig.add_shape(type="line", x0=min(y_pred_homo), y0=0, x1=max(y_pred_homo), y1=0,
                      line=dict(color="red", width=2))
        
        fig.update_layout(
            title="Homoscedastic Residuals",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Heteroscedastic Example")
        
        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                 marker=dict(color='blue', size=8, opacity=0.6),
                                 name='Residuals'))
        fig.add_shape(type="line", x0=min(y_pred), y0=0, x1=max(y_pred), y1=0,
                      line=dict(color="red", width=2))
        
        fig.update_layout(
            title="Heteroscedastic Residuals (Fan Pattern)",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("How to Fix Heteroscedasticity")
    
    fix_option = st.selectbox(
        "Select a method to fix heteroscedasticity:",
        ["Log Transformation of Dependent Variable", "Weighted Least Squares"]
    )
    
    if fix_option == "Log Transformation of Dependent Variable":
        # Apply log transformation
        df_transformed = df_hetero.copy()
        df_transformed['y_log'] = np.log1p(df_transformed['y'])  # log1p to handle zeros
        
        # Fit model on transformed data
        x_trans = df_transformed['x'].values
        y_trans = df_transformed['y_log'].values
        
        slope_trans, intercept_trans = np.polyfit(x_trans, y_trans, 1)
        y_pred_trans = slope_trans * x_trans + intercept_trans
        residuals_trans = y_trans - y_pred_trans
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred_trans, y=residuals_trans, mode='markers',
                                 marker=dict(color='green', size=8, opacity=0.6),
                                 name='Residuals after Log Transform'))
        fig.add_shape(type="line", x0=min(y_pred_trans), y0=0, x1=max(y_pred_trans), y1=0,
                      line=dict(color="red", width=2))
        
        fig.update_layout(
            title="Residuals After Log Transformation",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif fix_option == "Weighted Least Squares":
        st.markdown("""
        In Weighted Least Squares (WLS), we give different weights to observations based on their variance.
        
        1. First, we fit an OLS model
        2. Calculate absolute residuals
        3. Fit a model to predict these absolute residuals
        4. Use the reciprocal of squared fitted values as weights
        5. Fit the final WLS model
        """)
        
        # Simulate WLS results
        np.random.seed(42)
        weights = 1 / (0.5 + 0.1 * df_hetero['x'])
        weighted_residuals = np.random.normal(0, 1, len(df_hetero)) / np.sqrt(weights)
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=weighted_residuals, mode='markers',
                                 marker=dict(color='purple', size=8, opacity=0.6),
                                 name='Weighted Residuals'))
        fig.add_shape(type="line", x0=min(y_pred), y0=0, x1=max(y_pred), y1=0,
                      line=dict(color="red", width=2))
        
        fig.update_layout(
            title="Residuals After Weighted Least Squares",
            xaxis_title="Fitted Values",
            yaxis_title="Weighted Residuals",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

elif assumption == "Normality of Residuals":
    st.header("4. Normality of Residuals Assumption")
    
    st.markdown("""
    **Assumption**: Residuals should be normally distributed.
    
    **How to test**: Create Q-Q plots or histogram of residuals.
    """)
    
    # Fit linear model
    x = df_norm['x'].values
    y = df_norm['y'].values
    
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Histogram of Residuals")
        
        # Create histogram
        fig = px.histogram(
            x=residuals,
            nbins=20,
            marginal="box",
            title="Distribution of Residuals",
            labels={"x": "Residuals"}
        )
        
        # Add KDE curve
        hist_data = [residuals]
        group_labels = ['Residuals']
        
        fig2 = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
        fig.add_trace(fig2.data[0])
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Q-Q Plot")
        
        # Create Q-Q plot
        qq = stats.probplot(residuals, dist="norm")
        x_qq = qq[0][0]
        y_qq = qq[0][1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_qq,
            y=y_qq,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Q-Q Plot'
        ))
        
        # Add reference line
        min_val = min(x_qq.min(), y_qq.min())
        max_val = max(x_qq.max(), y_qq.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Reference Line'
        ))
        
        fig.update_layout(
            title="Q-Q Plot (Points should follow the line)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("How to Fix Non-Normal Residuals")
    
    # Modified to only show "Remove Outliers" option
    st.markdown("""
    **Remove Outliers**: Identifying and removing outliers can help normalize the distribution of residuals.
    """)
    
    # Generate data with outliers
    np.random.seed(42)
    x_out = np.random.uniform(0, 10, 200)
    y_out = 2 * x_out + np.random.normal(0, 1, 200)
    
    # Add outliers
    outlier_indices = np.random.choice(range(len(x_out)), 10, replace=False)
    y_out[outlier_indices] = y_out[outlier_indices] + np.random.uniform(5, 10, 10)
    
    # Fit model with outliers
    slope_out, intercept_out = np.polyfit(x_out, y_out, 1)
    y_pred_out = slope_out * x_out + intercept_out
    residuals_out = y_out - y_pred_out
    
    # Remove outliers (residuals more than 2 standard deviations)
    std_resid = np.std(residuals_out)
    outlier_mask = np.abs(residuals_out) > 2 * std_resid
    x_clean = x_out[~outlier_mask]
    y_clean = y_out[~outlier_mask]
    
    # Fit model without outliers
    slope_clean, intercept_clean = np.polyfit(x_clean, y_clean, 1)
    y_pred_clean = slope_clean * x_clean + intercept_clean
    residuals_clean = y_clean - y_pred_clean
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Q-Q plot with outliers
        qq_out = stats.probplot(residuals_out, dist="norm")
        x_qq_out = qq_out[0][0]
        y_qq_out = qq_out[0][1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_qq_out, y=y_qq_out, mode='markers', name='Residuals with Outliers'
        ))
        
        min_val = min(x_qq_out.min(), y_qq_out.min())
        max_val = max(x_qq_out.max(), y_qq_out.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='red', dash='dash'),
            name='Reference Line'
        ))
        
        fig.update_layout(
            title="Q-Q Plot with Outliers",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Q-Q plot without outliers
        qq_clean = stats.probplot(residuals_clean, dist="norm")
        x_qq_clean = qq_clean[0][0]
        y_qq_clean = qq_clean[0][1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_qq_clean, y=y_qq_clean, mode='markers', name='Residuals without Outliers'
        ))
        
        min_val = min(x_qq_clean.min(), y_qq_clean.min())
        max_val = max(x_qq_clean.max(), y_qq_clean.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='red', dash='dash'),
            name='Reference Line'
        ))
        
        fig.update_layout(
            title="Q-Q Plot After Removing Outliers",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Add footer with additional information
st.markdown("---")
st.markdown("""
### Additional Resources
- This interactive app demonstrates the four key assumptions of linear regression as shown in the reference image.
- Each section provides methods to test and fix violations of these assumptions.
- For more advanced analysis, consider using statistical libraries like statsmodels or scikit-learn.
""")
