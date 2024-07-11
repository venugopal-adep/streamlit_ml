import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Tree Pruning Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🌳 Decision Tree Pruning Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of tree pruning in reducing overfitting and improving model generalization!")

# Sidebar
st.sidebar.header("Pruning Settings")
prune_level = st.sidebar.slider("Select the pruning level:", min_value=1, max_value=10, value=3, step=1)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Tree Pruning in Action")
    
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the original decision tree classifier
    clf_original = DecisionTreeClassifier(random_state=42)
    clf_original.fit(X_train, y_train)
    
    # Train the pruned decision tree classifier
    clf_pruned = DecisionTreeClassifier(random_state=42, max_depth=prune_level)
    clf_pruned.fit(X_train, y_train)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Tree")
        fig_original, ax_original = plt.subplots(figsize=(10, 8))
        plot_tree(clf_original, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax_original)
        st.pyplot(fig_original)
        
        accuracy_original = accuracy_score(y_test, clf_original.predict(X_test))
        st.metric("Test Accuracy", f"{accuracy_original:.2f}")
        st.metric("Tree Depth", clf_original.get_depth())
        st.metric("Number of Nodes", clf_original.get_n_leaves())
    
    with col2:
        st.subheader("Pruned Tree")
        fig_pruned, ax_pruned = plt.subplots(figsize=(10, 8))
        plot_tree(clf_pruned, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax_pruned)
        st.pyplot(fig_pruned)
        
        accuracy_pruned = accuracy_score(y_test, clf_pruned.predict(X_test))
        st.metric("Test Accuracy", f"{accuracy_pruned:.2f}")
        st.metric("Tree Depth", clf_pruned.get_depth())
        st.metric("Number of Nodes", clf_pruned.get_n_leaves())

with tab2:
    st.header("Solved Example: Impact of Pruning")
    
    prune_levels = range(1, 11)
    accuracies = []
    
    for level in prune_levels:
        clf = DecisionTreeClassifier(random_state=42, max_depth=level)
        clf.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
    
    fig, ax = plt.subplots()
    ax.plot(prune_levels, accuracies, marker='o')
    ax.set_xlabel("Max Tree Depth")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Impact of Pruning on Test Accuracy")
    st.pyplot(fig)
    
    st.write("This example shows how the test accuracy changes as we increase the maximum depth of the tree.")
    st.write(f"The optimal depth for this dataset appears to be around {np.argmax(accuracies) + 1}, with a test accuracy of {max(accuracies):.2f}.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of tree pruning?",
            "options": ["Increase model complexity", "Reduce overfitting", "Increase training time", "Decrease interpretability"],
            "correct": 1,
            "explanation": "Tree pruning is primarily used to reduce overfitting. By limiting the depth or complexity of the tree, we can create a model that generalizes better to unseen data, even if it might slightly decrease performance on the training data."
        },
        {
            "question": "Which of the following is NOT a common method for tree pruning?",
            "options": ["Reduced Error Pruning", "Cost Complexity Pruning", "Depth-based Pruning", "Random Node Elimination"],
            "correct": 3,
            "explanation": "Common pruning methods include Reduced Error Pruning, Cost Complexity Pruning (also known as Weakest Link Pruning), and simple depth-based pruning. Random Node Elimination is not a standard pruning technique, as pruning is typically done in a systematic way to improve the model's performance and generalization."
        },
        {
            "question": "What is a potential drawback of excessive pruning?",
            "options": ["Increased overfitting", "Decreased interpretability", "Increased model complexity", "Underfitting"],
            "correct": 3,
            "explanation": "While pruning helps to reduce overfitting, excessive pruning can lead to underfitting. This happens when the model becomes too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data."
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
    st.header("Learn More About Tree Pruning")
    st.markdown("""
    Tree pruning is a technique used in decision tree algorithms to reduce the size of decision trees by removing sections of the tree that provide little power to classify instances. Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting.

    Key benefits of Tree Pruning:
    1. **Reduces Overfitting**: By simplifying the tree, pruning helps the model generalize better to unseen data.
    2. **Improves Interpretability**: Smaller trees are easier to understand and explain.
    3. **Reduces Computational Complexity**: Pruned trees require less memory and are faster to use for predictions.

    Common Pruning Techniques:
    1. **Pre-pruning (Early Stopping)**: Stop growing the tree earlier, before it perfectly classifies the training set.
    2. **Post-pruning**: Grow the full tree, then prune it back to find the subtree with the lowest test error rate.

    When to use Tree Pruning:
    - When your decision tree is overfitting the training data
    - When you need a more interpretable model
    - When you want to reduce the computational resources required for predictions

    Remember, while pruning is generally beneficial, the optimal level of pruning can vary depending on the specific dataset and problem. It's often useful to use techniques like cross-validation to find the best pruning strategy for your particular case.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the effect of tree pruning on decision tree models. Adjust the pruning level and explore the different tabs to learn more!")
