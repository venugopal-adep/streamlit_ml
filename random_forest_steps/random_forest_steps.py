import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Set page configuration
st.set_page_config(page_title="Random Forest: Steps Involved", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🌳 Random Forest: Steps Involved")
st.markdown("**Developed by: Venugopal Adep**")

st.markdown("""
This interactive application demonstrates the steps involved in the Random Forest algorithm.
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
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
@st.cache_resource
def train_model(n_trees, max_depth):
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

rf_classifier = train_model(n_trees, max_depth)

# Main content using tabs
tab1, tab2, tab3 = st.tabs(["🔍 Overview", "📊 Model Performance", "📈 Visualizations"])

with tab1:
    st.header("🔍 Random Forest Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Steps Involved:")
        steps = [
            "1. Selection of a random subsample of a given dataset.",
            "2. Using attribute selection indicators create a decision tree for each subsample and record the prediction outcome from each model.",
            "3. Applying the voting/averaging method over predicted outcomes of individual models.",
            "4. Considering the final results as the average value or most voted value."
        ]
        for step in steps:
            st.write(step)
    
    with col2:
        st.subheader("Key Concepts:")
        st.write("""
        - Random Forest is an ensemble learning method.
        - It combines multiple decision trees to make predictions.
        - It's robust against overfitting and can handle high-dimensional data.
        - The algorithm uses bagging (bootstrap aggregating) to create diverse subsets of the data.
        """)
    
    st.subheader("Current Model Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of trees", n_trees)
    col2.metric("Maximum depth", max_depth)
    col3.metric("Number of features", X.shape[1])

with tab2:
    st.header("📊 Model Performance")
    
    y_pred = rf_classifier.predict(X_test)
    accuracy = rf_classifier.score(X_test, y_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy")
        st.metric("Accuracy", f"{accuracy:.2f}")
        
        st.write("""
        Model accuracy represents the proportion of correct predictions (both true positives and true negatives) 
        among the total number of cases examined. An accuracy of 1.0 means perfect prediction, while 0.5 would be 
        no better than random guessing for a binary classification problem.
        """)
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = rf_classifier.feature_importances_
        fig = go.Figure(data=[go.Bar(x=[f'Feature {i+1}' for i in range(len(feature_importance))], y=feature_importance)])
        fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance')
        st.plotly_chart(fig)



with tab3:
    st.header("📈 Voting Process and Final Prediction")
    
    tree_predictions = [tree.predict(X_test) for tree in rf_classifier.estimators_]
    vote_counts = [sum(pred) for pred in zip(*tree_predictions)]
    
    # Calculate the final prediction
    final_prediction = np.argmax(vote_counts)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Voting Process", "Final Prediction"),
                        column_widths=[0.7, 0.3])
    
    # Voting Process
    fig.add_trace(
        go.Bar(x=[f'Class {i}' for i in range(2)], y=vote_counts, name="Votes"),
        row=1, col=1
    )
    
    # Final Prediction
    fig.add_trace(
        go.Bar(x=['Accuracy'], y=[accuracy], name="Accuracy",
               text=[f"{accuracy:.2f}"], textposition="outside"),
        row=1, col=2
    )
    
    # Add annotation for final prediction
    fig.add_annotation(
        x=0.5, y=1.05,
        xref="x domain", yref="y domain",
        text=f"Final Prediction: Class {final_prediction}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="#f0f8ff",
        bordercolor="#2c3e50",
        borderwidth=2,
        row=1, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Classes", row=1, col=1)
    fig.update_xaxes(title_text="Metric", row=1, col=2)
    fig.update_yaxes(title_text="Vote Count", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **Voting Process:** This chart shows the number of votes each class received from the individual trees in the Random Forest.
    The class with the most votes becomes the final prediction.
    
    **Final Prediction:** The class with the highest number of votes is chosen as the final prediction. This is shown above the Voting Process chart.
    
    **Accuracy:** This chart displays the overall accuracy of the model on the test set, which represents
    the proportion of correct predictions made by the Random Forest.
    """)



st.markdown("""
## 🎓 Conclusion

Congratulations on exploring the Random Forest algorithm! Remember:

- 🌳 Random Forest combines multiple decision trees for robust predictions.
- 🔢 Hyperparameters like the number of trees and maximum depth can significantly affect performance.
- 📊 Feature importance helps understand which variables are most influential in the model.
- 🗳️ The voting process aggregates predictions from individual trees to make the final decision.

Keep exploring and refining your understanding of machine learning algorithms!
""")
