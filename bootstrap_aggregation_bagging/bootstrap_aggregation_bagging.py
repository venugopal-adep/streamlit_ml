import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Bagging Classifier Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🌳 Bootstrap Aggregation (Bagging) Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of ensemble learning with this interactive Bagging Classifier demo!")

# Sidebar
st.sidebar.header("Classifier Settings")
n_classifiers = st.sidebar.slider("Number of classifiers:", min_value=1, max_value=10, value=3, step=1)
sample_size = st.sidebar.slider("Sample size (% of original data):", min_value=10, max_value=100, value=50, step=10)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Bagging Classifier in Action")
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    
    # Create a scatter plot of the original data
    fig_original = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                                              marker=dict(color=y, colorscale='Viridis', size=8))])
    fig_original.update_layout(title='Original Data', xaxis_title='Feature 1', yaxis_title='Feature 2',
                               template="plotly_white")
    st.plotly_chart(fig_original, use_container_width=True)
    
    # Bootstrap sampling and training individual classifiers
    classifiers = []
    col1, col2 = st.columns(2)
    
    for i in range(n_classifiers):
        indices = np.random.choice(len(X), size=int(len(X) * sample_size / 100), replace=True)
        X_subset, y_subset = X[indices], y[indices]
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_subset, y_subset)
        classifiers.append(clf)
        
        # Create a scatter plot for each classifier
        fig_classifier = go.Figure(data=[go.Scatter(x=X_subset[:, 0], y=X_subset[:, 1], mode='markers', 
                                                    marker=dict(color=y_subset, colorscale='Viridis', size=8))])
        fig_classifier.update_layout(title=f'Classifier {i+1}', xaxis_title='Feature 1', yaxis_title='Feature 2',
                                     template="plotly_white")
        
        if i % 2 == 0:
            with col1:
                st.plotly_chart(fig_classifier, use_container_width=True)
        else:
            with col2:
                st.plotly_chart(fig_classifier, use_container_width=True)
    
    # Aggregate the predictions of individual classifiers
    bagging_clf = BaggingClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=n_classifiers)
    bagging_clf.fit(X, y)
    
    # Make predictions using the bagging classifier
    y_pred = bagging_clf.predict(X)
    
    # Create a scatter plot of the bagging classifier's predictions
    fig_bagging = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                                             marker=dict(color=y_pred, colorscale='Viridis', size=8))])
    fig_bagging.update_layout(title='Bagging Classifier Predictions', xaxis_title='Feature 1', yaxis_title='Feature 2',
                              template="plotly_white")
    st.plotly_chart(fig_bagging, use_container_width=True)
    
    # Evaluate the bagging classifier
    accuracy = accuracy_score(y, y_pred)
    st.metric("Bagging Classifier Accuracy", f"{accuracy:.2f}")

with tab2:
    st.header("Solved Example: Iris Dataset")
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    # Train a bagging classifier
    bagging_clf_iris = BaggingClassifier(n_estimators=10, random_state=42)
    bagging_clf_iris.fit(X_train, y_train)
    
    # Make predictions
    y_pred_iris = bagging_clf_iris.predict(X_test)
    
    # Calculate accuracy
    accuracy_iris = accuracy_score(y_test, y_pred_iris)
    
    st.write(f"We trained a Bagging Classifier on the Iris dataset with 10 base estimators.")
    st.write(f"The accuracy on the test set is: {accuracy_iris:.2f}")
    
    # Visualize feature importance
    feature_importance = np.mean([tree.feature_importances_ for tree in bagging_clf_iris.estimators_], axis=0)
    
    fig, ax = plt.subplots()
    ax.bar(iris.feature_names, feature_importance)
    ax.set_title("Feature Importance in Bagging Classifier")
    ax.set_ylabel("Importance")
    st.pyplot(fig)

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main advantage of using Bagging?",
            "options": ["Increased model complexity", "Reduced overfitting", "Faster training time", "Smaller model size"],
            "correct": 1,
            "explanation": "Bagging helps reduce overfitting by training multiple models on different subsets of the data. This ensemble approach helps to average out the individual models' errors, resulting in a more robust and generalizable final model."
        },
        {
            "question": "In Bagging, how are the subsets of data chosen for each base estimator?",
            "options": ["Randomly with replacement", "Randomly without replacement", "Sequentially", "Based on feature importance"],
            "correct": 0,
            "explanation": "In Bagging, subsets are chosen randomly with replacement. This means that some samples may appear multiple times in a subset, while others may not appear at all. This process, known as bootstrap sampling, introduces variability among the base estimators."
        },
        {
            "question": "What is typically used as the base estimator in a Random Forest?",
            "options": ["Support Vector Machine", "Logistic Regression", "Decision Tree", "Neural Network"],
            "correct": 2,
            "explanation": "Random Forest is a specific type of Bagging where the base estimators are Decision Trees. Each tree in the forest is trained on a bootstrap sample of the data, and typically considers only a random subset of features when splitting nodes."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
            st.write(f"Explanation: {q['explanation']}")
        st.write("---")

with tab4:
    st.header("Learn More About Bagging")
    st.markdown("""
    Bagging, short for Bootstrap Aggregating, is an ensemble learning technique that combines multiple models to improve prediction accuracy and reduce overfitting. Here's how it works:

    1. **Bootstrap Sampling**: Create multiple subsets of the original dataset by randomly sampling with replacement.
    2. **Model Training**: Train a separate model on each subset.
    3. **Aggregation**: Combine the predictions of all models (e.g., by voting for classification or averaging for regression).

    Key benefits of Bagging:
    - Reduces overfitting
    - Improves stability and accuracy
    - Works well with high-variance, low-bias models (e.g., decision trees)

    Popular implementations:
    - Random Forests (Bagging with Decision Trees)
    - Extra Trees

    When to use Bagging:
    - When you have a complex dataset prone to overfitting
    - When you want to improve model stability and reduce variance
    - When you have computational resources to train multiple models

    Remember, while Bagging is powerful, it's not always the best choice. Consider other ensemble methods like Boosting or Stacking depending on your specific problem and dataset characteristics.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the power of Bagging in machine learning. Adjust the settings and explore the different tabs to learn more!")
