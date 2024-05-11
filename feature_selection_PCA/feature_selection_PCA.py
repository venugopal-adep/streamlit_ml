import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Set page title
st.set_page_config(page_title="Feature Selection Demo")

# Title and description
st.write("## Feature Selection Demo")
st.write('**Developed by : Venugopal Adep**')
st.markdown("This application demonstrates feature selection in machine learning")

# Dataset selection
dataset = st.sidebar.selectbox("Select a dataset", ("Iris", "Wine"))

# Load the selected dataset
if dataset == "Iris":
    data = load_iris()
else:
    data = load_wine()

X = data.data
y = data.target

# Create a DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Feature selection
st.sidebar.subheader("Feature Selection")
k = st.sidebar.slider("Number of top features", 1, len(data.feature_names), len(data.feature_names))

selector = SelectKBest(f_classif, k=k)
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = [data.feature_names[i] for i in selector.get_support(indices=True)]

# PCA for visualization
st.sidebar.subheader("PCA")
n_components = st.sidebar.radio("Number of principal components", (2, 3))

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_selected)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['target'] = y

# Assign colors to selected features
color_discrete_sequence = px.colors.qualitative.Plotly
feature_colors = {feature: color_discrete_sequence[i % len(color_discrete_sequence)]
                  for i, feature in enumerate(selected_features)}
df_pca['Feature'] = df_pca.index.map(lambda x: selected_features[x % len(selected_features)])
df_pca['Color'] = df_pca['Feature'].map(feature_colors)

# Plotly scatter plot
if n_components == 2:
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Feature', labels={'color': 'Feature'},
                     title=f"PCA Visualization of Selected Features",
                     color_discrete_sequence=list(feature_colors.values()))
else:
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Feature', labels={'color': 'Feature'},
                        title=f"PCA Visualization of Selected Features",
                        color_discrete_sequence=list(feature_colors.values()))

# Display the plot
st.plotly_chart(fig)

# Bar plot of feature importance
st.subheader("Feature Importance")
feature_importance = selector.scores_
fig_importance = go.Figure(data=[go.Bar(x=selected_features, y=feature_importance[selector.get_support(indices=True)],
                                         marker_color=[feature_colors[feature] for feature in selected_features])])
fig_importance.update_layout(xaxis_title="Feature", yaxis_title="Importance Score")
st.plotly_chart(fig_importance)

# Bar plot of principal component importance
st.subheader("Principal Component Importance")
pc_importance = pca.explained_variance_ratio_
fig_pc_importance = go.Figure(data=[go.Bar(x=[f'PC{i+1}' for i in range(n_components)], y=pc_importance,
                                            marker_color=color_discrete_sequence[:n_components])])
fig_pc_importance.update_layout(xaxis_title="Principal Component", yaxis_title="Importance Score")
st.plotly_chart(fig_pc_importance)

# Brief writeup on feature selection
st.subheader("How Feature Selection Works")
st.markdown("""
Feature selection is the process of selecting a subset of relevant features (variables) from a larger set of features in a dataset. The goal is to improve model performance, reduce computational complexity, and enhance interpretability by focusing on the most informative features.

There are different approaches to feature selection, including:

1. Filter methods: These methods select features based on statistical measures such as correlation, chi-squared test, or information gain. The features are ranked according to their relevance, and a subset of top-ranked features is selected.

2. Wrapper methods: These methods evaluate subsets of features by training and testing a specific machine learning model. The subset that yields the best model performance is selected. Examples include recursive feature elimination (RFE) and forward/backward feature selection.

3. Embedded methods: These methods perform feature selection as part of the model training process. The model itself is responsible for selecting the most relevant features. Examples include L1 regularization (Lasso) and decision tree-based feature importance.

In this demo, we use the SelectKBest method from scikit-learn, which is a filter method. It selects the top k features based on a scoring function, such as the F-value between the feature and the target variable. The user can interactively choose the number of top features to select using the slider.

The selected features are then visualized using Principal Component Analysis (PCA) to reduce the dimensionality and provide a visual representation of the data. The importance of each selected feature and the principal components are displayed using bar plots.

By applying feature selection, we can identify the most informative features that contribute to the target variable and potentially improve the performance and interpretability of machine learning models.
""")