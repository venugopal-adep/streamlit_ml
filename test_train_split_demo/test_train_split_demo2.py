import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Set page config
st.set_page_config(page_title="Train-Test Split Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🔀 Train-Test Split Interactive Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the impact of train-test split on machine learning model performance!")

# Helper functions
def load_dataset(dataset_name):
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer()
    }
    return datasets[dataset_name]

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    return model, y_pred, accuracy, recall, precision

# Sidebar
st.sidebar.header("Configuration")
dataset_name = st.sidebar.selectbox('Select Dataset', ['Iris', 'Wine', 'Breast Cancer'])
test_size = st.sidebar.slider('Test Set Size (%)', min_value=10, max_value=50, value=25, step=5) / 100.0
resample_data = st.sidebar.button('Resample')

# Load data
data = load_dataset(dataset_name)
X, y = data['data'], data['target']
feature_names = data['feature_names']
class_names = data['target_names']

# Split the data
if resample_data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train and evaluate model
model, y_pred, accuracy, recall, precision = train_and_evaluate_model(X_train, X_test, y_train, y_test)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualization", "📈 Model Performance", "🧠 Confusion Matrix", "📚 Learn More"])

with tab1:
    st.header("Train-Test Split Visualization")
    
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['type'] = 'Train'
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['type'] = 'Test'
    df_combined = pd.concat([df_train, df_test])
    
    colors = {'Train': 'red', 'Test': 'green'}
    fig = px.scatter_3d(df_combined, x=feature_names[0], y=feature_names[1], z=feature_names[2], color='type', 
                        color_discrete_map=colors, title='Train-Test Split 3D Visualization')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
    with col2:
        st.metric("Recall", f"{recall:.2f}")
    with col3:
        st.metric("Precision", f"{precision:.2f}")
    
    # Feature importance (for KNN, we'll use a simple metric based on the nearest neighbors)
    distances, indices = model.kneighbors(X_test)
    feature_importance = np.mean(np.abs(X_test[:, None, :] - X_train[indices]), axis=(0, 1))
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    fig_importance = px.bar(feature_importance_df, x='feature', y='importance', title='Feature Importance')
    st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.header("Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig_cm = px.imshow(cm_df, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"), 
                       color_continuous_scale=px.colors.sequential.Viridis)
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with tab4:
    st.header("Learn More About Train-Test Split")
    st.markdown("""
    The train-test split is a technique used in machine learning to evaluate the performance of a model on unseen data. Here are some key points:

    1. **Purpose**: It helps assess how well a model generalizes to new, unseen data.
    2. **Process**: The dataset is divided into two subsets:
        - Training set: Used to train the model
        - Test set: Used to evaluate the model's performance
    3. **Typical split ratio**: Common ratios include 80-20 or 70-30 (train-test).
    4. **Randomization**: Data is usually randomly split to ensure both sets are representative of the overall dataset.
    5. **Stratification**: For classification problems, stratified sampling ensures that the class distribution is preserved in both sets.
    6. **Trade-offs**: 
        - Larger training set: More data for the model to learn from, but less data for testing.
        - Larger test set: More reliable performance estimates, but less data for training.
    7. **Limitations**: 
        - Doesn't work well with small datasets
        - May not capture temporal aspects in time-series data
    8. **Alternatives**: Cross-validation, especially for smaller datasets.

    Remember, the goal is to create a model that performs well on unseen data, not just memorize the training data!
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the impact of train-test split on model performance. Adjust the test size, resample the data, and explore the different tabs to learn more!")
