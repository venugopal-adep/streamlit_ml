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
.highlight {
    background-color: #ffff00;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔍 Feature Selection and PCA Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of feature selection and dimensionality reduction in machine learning!")

# Helper functions
@st.cache_data
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Visualization", "📈 Feature Importance", "🧠 PCA Analysis", "🧠 Quiz"])

with tab1:
    st.header("Understanding Feature Selection and PCA")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Feature Selection?</h3>
    <p>Feature selection is the process of selecting a subset of relevant features from a larger set of features in a dataset. The main goals are to improve model performance, reduce computational complexity, and enhance interpretability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Types of Feature Selection</h3>
    <ul>
        <li><strong>Filter methods:</strong> Select features based on statistical measures</li>
        <li><strong>Wrapper methods:</strong> Evaluate subsets of features by training and testing a specific model</li>
        <li><strong>Embedded methods:</strong> Perform feature selection as part of the model training process</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>What is Principal Component Analysis (PCA)?</h3>
    <p>PCA is a dimensionality reduction technique that finds the directions of maximum variance in the data and projects it onto a lower-dimensional subspace.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why are Feature Selection and PCA Important?</h3>
    <ul>
        <li><span class="highlight">Improved Model Performance:</span> By focusing on the most relevant features</li>
        <li><span class="highlight">Reduced Overfitting:</span> Fewer features can lead to more generalizable models</li>
        <li><span class="highlight">Faster Training:</span> Fewer features mean less computational complexity</li>
        <li><span class="highlight">Better Interpretability:</span> Understanding which features are most important</li>
        <li><span class="highlight">Visualization:</span> PCA allows us to visualize high-dimensional data in 2D or 3D</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
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

with tab3:
    st.header("Feature Importance")
    
    feature_importance = selector.scores_
    fig_importance = go.Figure(data=[go.Bar(x=selected_features, y=feature_importance[selector.get_support(indices=True)],
                                            marker_color=[feature_colors[feature] for feature in selected_features])])
    fig_importance.update_layout(xaxis_title="Feature", yaxis_title="Importance Score")
    st.plotly_chart(fig_importance, use_container_width=True)

with tab4:
    st.header("Principal Component Analysis")
    
    pc_importance = pca.explained_variance_ratio_
    fig_pc_importance = go.Figure(data=[go.Bar(x=[f'PC{i+1}' for i in range(n_components)], y=pc_importance,
                                               marker_color=color_discrete_sequence[:n_components])])
    fig_pc_importance.update_layout(xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio")
    st.plotly_chart(fig_pc_importance, use_container_width=True)

with tab5:
    st.header("Test Your Knowledge")

    questions = [
        {
            "question": "What is the main goal of feature selection?",
            "options": ["To increase the number of features", "To select the most relevant features", "To create new features"],
            "correct": 1,
            "explanation": "Feature selection aims to select the most relevant features to improve model performance and reduce complexity."
        },
        {
            "question": "What does PCA stand for?",
            "options": ["Principal Component Analysis", "Potential Component Algorithm", "Predictive Classification Approach"],
            "correct": 0,
            "explanation": "PCA stands for Principal Component Analysis, which is a dimensionality reduction technique."
        },
        {
            "question": "Which of the following is NOT a benefit of feature selection?",
            "options": ["Improved model performance", "Reduced overfitting", "Increased computational complexity"],
            "correct": 2,
            "explanation": "Feature selection typically reduces computational complexity, not increases it."
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
st.sidebar.info("This app demonstrates feature selection and PCA techniques. Select a dataset, adjust the number of features and principal components, then explore the different tabs to learn more!")
