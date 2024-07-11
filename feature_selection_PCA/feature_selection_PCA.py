import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Feature Selection Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🔍 Feature Selection Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of feature selection in machine learning!")

# Helper functions
def load_dataset(dataset_name):
    if dataset_name == "Iris":
        return load_iris()
    else:
        return load_wine()

def perform_feature_selection(X, y, k):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = [data.feature_names[i] for i in selector.get_support(indices=True)]
    return X_selected, selected_features, selector

def perform_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Sidebar
st.sidebar.header("Configuration")
dataset_name = st.sidebar.selectbox("Select a dataset", ("Iris", "Wine"))

# Load data
data = load_dataset(dataset_name)
X, y = data.data, data.target

# Dynamically set the maximum value for the slider
max_features = X.shape[1]
k = st.sidebar.slider("Number of top features", 1, max_features, max_features)

n_components = st.sidebar.radio("Number of principal components", (2, 3))

# Perform feature selection and PCA
X_selected, selected_features, selector = perform_feature_selection(X, y, k)
X_pca, pca = perform_pca(X_selected, n_components)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualization", "📈 Feature Importance", "🧠 PCA Analysis", "📚 Learn More"])

with tab1:
    st.header("Data Visualization")
    
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca['target'] = y
    df_pca['Feature'] = df_pca.index.map(lambda x: selected_features[x % len(selected_features)])
    
    color_discrete_sequence = px.colors.qualitative.Plotly
    feature_colors = {feature: color_discrete_sequence[i % len(color_discrete_sequence)]
                      for i, feature in enumerate(selected_features)}
    df_pca['Color'] = df_pca['Feature'].map(feature_colors)
    
    if n_components == 2:
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Feature', labels={'color': 'Feature'},
                         title=f"PCA Visualization of Selected Features",
                         color_discrete_sequence=list(feature_colors.values()))
    else:
        fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Feature', labels={'color': 'Feature'},
                            title=f"PCA Visualization of Selected Features",
                            color_discrete_sequence=list(feature_colors.values()))
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Feature Importance")
    
    feature_importance = selector.scores_
    fig_importance = go.Figure(data=[go.Bar(x=selected_features, y=feature_importance[selector.get_support(indices=True)],
                                            marker_color=[feature_colors[feature] for feature in selected_features])])
    fig_importance.update_layout(xaxis_title="Feature", yaxis_title="Importance Score")
    st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.header("Principal Component Analysis")
    
    pc_importance = pca.explained_variance_ratio_
    fig_pc_importance = go.Figure(data=[go.Bar(x=[f'PC{i+1}' for i in range(n_components)], y=pc_importance,
                                               marker_color=color_discrete_sequence[:n_components])])
    fig_pc_importance.update_layout(xaxis_title="Principal Component", yaxis_title="Importance Score")
    st.plotly_chart(fig_pc_importance, use_container_width=True)

with tab4:
    st.header("Learn More About Feature Selection")
    st.markdown("""
    Feature selection is the process of selecting a subset of relevant features from a larger set of features in a dataset. The main goals of feature selection are:

    1. Improve model performance
    2. Reduce computational complexity
    3. Enhance interpretability

    There are three main approaches to feature selection:

    1. **Filter methods**: Select features based on statistical measures (e.g., correlation, chi-squared test, information gain).
    2. **Wrapper methods**: Evaluate subsets of features by training and testing a specific machine learning model (e.g., recursive feature elimination, forward/backward selection).
    3. **Embedded methods**: Perform feature selection as part of the model training process (e.g., L1 regularization, decision tree-based importance).

    This demo uses the SelectKBest method, which is a filter method. It selects the top k features based on a scoring function (F-value between feature and target variable).

    Principal Component Analysis (PCA) is used to visualize the selected features by reducing dimensionality. PCA finds the directions of maximum variance in the data and projects it onto a lower-dimensional subspace.

    By applying feature selection, we can identify the most informative features that contribute to the target variable and potentially improve the performance and interpretability of machine learning models.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates feature selection techniques. Select a dataset, adjust the number of features, and explore the different tabs to learn more!")
