import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

def plot_3d_split(df_combined, feature_names):
    colors = {'Train': 'red', 'Test': 'green'}
    fig = px.scatter_3d(df_combined, x=feature_names[0], y=feature_names[1], z=feature_names[2], color='type', 
                        color_discrete_map=colors, title='Train-Test Split 3D Visualization')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=600)
    return fig

def plot_feature_importance(feature_names, feature_importance):
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    fig = px.bar(feature_importance_df, x='feature', y='importance', title='Feature Importance')
    return fig

def plot_confusion_matrix(cm, class_names):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = px.imshow(cm_df, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"), 
                    color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(title="Confusion Matrix")
    return fig

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
if resample_data or 'X_train' not in st.session_state:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
    st.session_state.X_train, st.session_state.X_test = X_train, X_test
    st.session_state.y_train, st.session_state.y_test = y_train, y_test
else:
    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    y_train, y_test = st.session_state.y_train, st.session_state.y_test

# Train and evaluate model
model, y_pred, accuracy, recall, precision = train_and_evaluate_model(X_train, X_test, y_train, y_test)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Visualization", "📈 Model Performance", "🧠 Confusion Matrix", "🧠 Quiz"])

with tab1:
    st.header("Understanding Train-Test Split")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Train-Test Split?</h3>
    <p>Train-test split is a technique used in machine learning to evaluate the performance of a model on unseen data. It involves dividing the dataset into two subsets: one for training the model and another for testing its performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts</h3>
    <ul>
        <li><strong>Training Set:</strong> The subset of data used to train the model.</li>
        <li><strong>Test Set:</strong> The subset of data used to evaluate the model's performance.</li>
        <li><strong>Split Ratio:</strong> The proportion of data allocated to training and testing (e.g., 80-20 or 70-30).</li>
        <li><strong>Randomization:</strong> Ensures both sets are representative of the overall dataset.</li>
        <li><strong>Stratification:</strong> Preserves the class distribution in both sets for classification problems.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Train-Test Split?</h3>
    <ul>
        <li><span class="highlight">Model Evaluation:</span> Assess how well the model generalizes to new, unseen data.</li>
        <li><span class="highlight">Prevent Overfitting:</span> Helps detect if the model is memorizing the training data instead of learning general patterns.</li>
        <li><span class="highlight">Performance Estimation:</span> Provides a more realistic estimate of the model's performance on new data.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Train-Test Split Visualization")
    
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['type'] = 'Train'
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['type'] = 'Test'
    df_combined = pd.concat([df_train, df_test])
    
    fig = plot_3d_split(df_combined, feature_names)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
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
    
    fig_importance = plot_feature_importance(feature_names, feature_importance)
    st.plotly_chart(fig_importance, use_container_width=True)

with tab4:
    st.header("Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plot_confusion_matrix(cm, class_names)
    st.plotly_chart(fig_cm, use_container_width=True)

with tab5:
    st.header("Test Your Knowledge")

    questions = [
        {
            "question": "What is the main purpose of train-test split?",
            "options": ["To increase the dataset size", "To evaluate model performance on unseen data", "To speed up model training"],
            "correct": 1,
            "explanation": "The main purpose of train-test split is to evaluate how well a model performs on unseen data, helping to assess its generalization capability."
        },
        {
            "question": "What does a larger test set typically provide?",
            "options": ["Faster model training", "More reliable performance estimates", "Better model accuracy"],
            "correct": 1,
            "explanation": "A larger test set typically provides more reliable performance estimates, as it gives a better representation of how the model might perform on new, unseen data."
        },
        {
            "question": "What is stratification in train-test split?",
            "options": ["Splitting data randomly", "Preserving class distribution in both sets", "Increasing the dataset size"],
            "correct": 1,
            "explanation": "Stratification in train-test split ensures that the class distribution (proportion of samples for each class) is approximately the same in both the training and test sets."
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
st.sidebar.info("This app demonstrates the impact of train-test split on model performance. Adjust the test size, resample the data, and explore the different tabs to learn more!")
