import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Set page configuration
st.set_page_config(page_title="Random Forest Explorer", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stTab {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🌳 Random Forest Explorer")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
This interactive application demonstrates the Random Forest algorithm.
Explore different aspects of the algorithm and see how it works on a sample dataset.
""")

# Sidebar for hyperparameters
st.sidebar.header("Hyperparameters")
n_trees = st.sidebar.slider("Number of trees:", min_value=1, max_value=100, value=10, step=1)
max_depth = st.sidebar.slider("Maximum depth of trees:", min_value=1, max_value=10, value=5, step=1)

st.sidebar.write("""
**Number of trees:** The number of decision trees in the forest. More trees can lead to better performance but increase computation time.

**Maximum depth:** The maximum depth of each decision tree. Deeper trees can capture more complex patterns but may lead to overfitting.
""")

# Generate sample data
@st.cache_data
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = generate_data()

# Train the model
@st.cache_resource
def train_model(n_trees, max_depth):
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

rf_classifier = train_model(n_trees, max_depth)

# Main content using tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "🔍 Overview", "📊 Model Performance", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About Random Forest")
    
    st.markdown("""
    <div class="highlight">
    <h3>What is Random Forest?</h3>
    <p>Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. 
    It's like asking a group of experts (trees) for their opinion and then taking a vote to make the final decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>How Does Random Forest Work?</h3>
    <ol>
        <li>Create multiple subsets of the original dataset (bootstrap samples).</li>
        <li>Build a decision tree for each subset, using a random selection of features at each split.</li>
        <li>Each tree makes its own prediction.</li>
        <li>The final prediction is determined by majority vote (for classification) or averaging (for regression).</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why Use Random Forest?</h3>
    <ul>
        <li>Handles large datasets with higher dimensionality</li>
        <li>Reduces overfitting by averaging multiple trees</li>
        <li>Can handle missing values and maintain accuracy</li>
        <li>Provides feature importance rankings</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("🔍 Random Forest Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Steps Involved:")
        steps = [
            "1. Selection of random subsamples (with replacement) from the dataset.",
            "2. Creation of a decision tree for each subsample, using a random subset of features at each split.",
            "3. Collection of predictions from all trees.",
            "4. Final prediction by majority voting (classification) or averaging (regression)."
        ]
        for step in steps:
            st.write(step)
    
    with col2:
        st.subheader("Key Concepts:")
        st.write("""
        - Ensemble Learning: Combining multiple models for better predictions.
        - Bagging (Bootstrap Aggregating): Creating diverse subsets of the data.
        - Feature Randomness: Using a random subset of features for each split.
        - Voting/Averaging: Aggregating predictions from all trees.
        """)
    
    st.subheader("Current Model Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of trees", n_trees)
    col2.metric("Maximum depth", max_depth)
    col3.metric("Number of features", X_train.shape[1])

with tab3:
    st.header("📊 Model Performance")
    
    y_pred = rf_classifier.predict(X_test)
    accuracy = rf_classifier.score(X_test, y_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = accuracy,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Accuracy"},
            gauge = {'axis': {'range': [0, 1]},
                     'bar': {'color': "#1E90FF"},
                     'steps' : [
                         {'range': [0, 0.5], 'color': "lightgray"},
                         {'range': [0.5, 0.7], 'color': "gray"},
                         {'range': [0.7, 1], 'color': "darkgray"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': accuracy}}))
        st.plotly_chart(fig)
        
        st.write("""
        Model accuracy represents the proportion of correct predictions among the total number of cases examined. 
        A score of 1.0 means perfect prediction, while 0.5 would be no better than random guessing for a binary classification problem.
        """)
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = rf_classifier.feature_importances_
        fig = go.Figure(data=[go.Bar(
            x=feature_importance, 
            y=[f'Feature {i+1}' for i in range(len(feature_importance))],
            orientation='h'
        )])
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=400
        )
        st.plotly_chart(fig)
        
        st.write("""
        Feature importance shows how much each feature contributes to the predictions of the Random Forest. 
        Higher values indicate more important features.
        """)

with tab4:
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main advantage of Random Forest over a single decision tree?",
            "options": ["It's faster", "It reduces overfitting", "It uses less memory", "It's easier to interpret"],
            "correct": 1,
            "explanation": "Random Forest reduces overfitting by averaging predictions from multiple trees, making it more robust than a single decision tree."
        },
        {
            "question": "How does Random Forest make its final prediction for classification tasks?",
            "options": ["By averaging all tree predictions", "By majority voting", "By choosing the prediction of the deepest tree", "By random selection"],
            "correct": 1,
            "explanation": "For classification tasks, Random Forest uses majority voting, where the class predicted by the majority of the trees becomes the final prediction."
        },
        {
            "question": "What does 'feature importance' in Random Forest tell us?",
            "options": ["The order in which features were added to the model", "How often each feature is used in the trees", "Which features are easiest to collect", "The predictive power of each feature"],
            "correct": 3,
            "explanation": "Feature importance in Random Forest indicates the predictive power of each feature, showing which features contribute most to the model's predictions."
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

st.markdown("""
## 🎓 Conclusion

Great job exploring the Random Forest algorithm! Remember:

- 🌳 Random Forest combines multiple decision trees for robust predictions.
- 🔢 Hyperparameters like the number of trees and maximum depth can significantly affect performance.
- 📊 Feature importance helps understand which variables are most influential in the model.
- 🗳️ The voting process aggregates predictions from individual trees to make the final decision.

Keep exploring and refining your understanding of machine learning algorithms!
""")
