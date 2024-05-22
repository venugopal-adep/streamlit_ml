import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Tree Pruning Demo")

# Title and description
st.title("Tree Pruning Demo")
st.write("**Developed by : Venugopal Adep**")
st.write("This interactive application demonstrates the concept of Tree Pruning, which is the process of reducing the size of a decision tree by turning some branch nodes into leaf nodes and removing the leaf nodes under the original branch.")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Create the initial tree visualization
fig, ax = plt.subplots(figsize=(10, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
st.pyplot(fig)

# Pruning options
st.subheader("Pruning Options")
prune_level = st.sidebar.slider("Select the pruning level:", min_value=0, max_value=clf.tree_.max_depth, step=1, value=0)

# Perform pruning
clf_pruned = DecisionTreeClassifier(random_state=42, max_depth=prune_level)
clf_pruned.fit(X, y)

# Create the pruned tree visualization
fig_pruned, ax_pruned = plt.subplots(figsize=(10, 8))
plot_tree(clf_pruned, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax_pruned)
st.pyplot(fig_pruned)

# Evaluation metrics
st.subheader("Evaluation Metrics")
col1, col2 = st.columns(2)

with col1:
    st.write("Original Tree")
    st.write("Depth:", clf.tree_.max_depth)
    st.write("Nodes:", clf.tree_.node_count)
    st.write("Accuracy:", clf.score(X, y))

with col2:
    st.write("Pruned Tree")
    st.write("Depth:", clf_pruned.tree_.max_depth)
    st.write("Nodes:", clf_pruned.tree_.node_count)
    st.write("Accuracy:", clf_pruned.score(X, y))

# Explanation
st.subheader("Explanation")
st.write("In this application, you can adjust the pruning level using the slider to control the maximum depth of the decision tree. Pruning reduces the size of the tree by limiting its depth, which helps in reducing overfitting.")
st.write("As you increase the pruning level, the tree becomes smaller and simpler. The evaluation metrics show the impact of pruning on the tree's depth, number of nodes, and accuracy.")
st.write("A pruned tree with a smaller depth and fewer nodes is easier to interpret and may generalize better to new data, although it may have slightly lower accuracy on the training data compared to the original tree.")
