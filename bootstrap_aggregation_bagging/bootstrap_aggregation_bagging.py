import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Bagging Classifier Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🌳 Bootstrap Aggregation (Bagging) Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of ensemble learning with this interactive Bagging Classifier demo!")

# Sidebar
st.sidebar.header("Classifier Settings")
n_classifiers = st.sidebar.slider("Number of classifiers:", min_value=1, max_value=10, value=3, step=1)
sample_size = st.sidebar.slider("Sample size (% of original data):", min_value=10, max_value=100, value=50, step=10)

# Helper functions
@st.cache_data
def generate_data():
    X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    return X, y

def train_bagging_classifier(X, y, n_classifiers, sample_size):
    bagging_clf = BaggingClassifier(
        estimator=SVC(kernel='linear', probability=True),
        n_estimators=n_classifiers,
        max_samples=sample_size/100,
        random_state=42
    )
    bagging_clf.fit(X, y)
    return bagging_clf

def plot_scatter(X, y, title):
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y, title=title,
                     labels={'x': 'Feature 1', 'y': 'Feature 2'},
                     color_continuous_scale='viridis')
    fig.update_layout(template="plotly_white")
    return fig

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualization", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("Learn About Bagging")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3>What is Bagging?</h3>
    <p>Bagging, short for Bootstrap Aggregating, is like asking multiple experts for their opinion and then taking a vote. 
    Each 'expert' (or model) looks at a slightly different version of the data and makes its own decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>How Does Bagging Work?</h3>
    <ol>
        <li><span class="highlight">Bootstrap Sampling:</span> Create multiple subsets of the original dataset by randomly sampling with replacement.</li>
        <li><span class="highlight">Model Training:</span> Train a separate model on each subset.</li>
        <li><span class="highlight">Aggregation:</span> Combine the predictions of all models (e.g., by voting for classification or averaging for regression).</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Bagging?</h3>
    <ul>
        <li><span class="highlight">Reduces Overfitting:</span> By combining multiple models, bagging helps to average out individual errors.</li>
        <li><span class="highlight">Improves Stability:</span> The ensemble is less affected by small changes in the data.</li>
        <li><span class="highlight">Increases Accuracy:</span> Often performs better than individual models.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Bagging Classifier in Action")
    
    X, y = generate_data()
    
    st.plotly_chart(plot_scatter(X, y, "Original Data"), use_container_width=True)
    
    bagging_clf = train_bagging_classifier(X, y, n_classifiers, sample_size)
    
    y_pred = bagging_clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    st.plotly_chart(plot_scatter(X, y_pred, "Bagging Classifier Predictions"), use_container_width=True)
    
    st.metric("Bagging Classifier Accuracy", f"{accuracy:.2f}")
    
    if st.button("Show Individual Classifiers"):
        col1, col2 = st.columns(2)
        for i, estimator in enumerate(bagging_clf.estimators_):
            y_pred_single = estimator.predict(X)
            fig = plot_scatter(X, y_pred_single, f"Classifier {i+1}")
            if i % 2 == 0:
                col1.plotly_chart(fig, use_container_width=True)
            else:
                col2.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Solved Example: Iris Dataset")
    
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    bagging_clf_iris = BaggingClassifier(n_estimators=10, random_state=42)
    bagging_clf_iris.fit(X_train, y_train)
    
    y_pred_iris = bagging_clf_iris.predict(X_test)
    accuracy_iris = accuracy_score(y_test, y_pred_iris)
    
    st.write("We trained a Bagging Classifier on the Iris dataset with 10 base estimators.")
    st.metric("Test Accuracy", f"{accuracy_iris:.2f}")
    
    feature_importance = np.mean([tree.feature_importances_ for tree in bagging_clf_iris.estimators_], axis=0)
    
    fig = px.bar(x=iris.feature_names, y=feature_importance, 
                 labels={'x': 'Features', 'y': 'Importance'},
                 title="Feature Importance in Bagging Classifier")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does Bagging stand for?",
            "options": ["Boosting Aggregation", "Bootstrap Aggregating", "Best Aggregation", "Balanced Aggregating"],
            "correct": 1,
            "explanation": "Bagging stands for Bootstrap Aggregating. It involves creating multiple subsets of the data through bootstrap sampling and aggregating the results of models trained on these subsets."
        },
        {
            "question": "How does Bagging help improve model performance?",
            "options": ["By making the model more complex", "By reducing overfitting", "By increasing training time", "By using only the best features"],
            "correct": 1,
            "explanation": "Bagging helps improve model performance by reducing overfitting. It does this by training multiple models on different subsets of the data and combining their predictions."
        },
        {
            "question": "In Bagging, how are the final predictions made?",
            "options": ["By choosing the best model", "By averaging all model predictions", "By using only the first model", "By using the most recent model"],
            "correct": 1,
            "explanation": "In Bagging, the final predictions are typically made by averaging all model predictions (for regression) or by majority voting (for classification). This helps to reduce the impact of individual model errors."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Not quite right. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the power of Bagging in machine learning. Adjust the settings and explore the different tabs to learn more!")
