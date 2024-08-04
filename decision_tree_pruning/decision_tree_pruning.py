import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(page_title="Tree Pruning Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🌳 Decision Tree Pruning Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of tree pruning in reducing overfitting and improving model generalization!")

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y, iris.target_names

X, y, target_names = load_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar
st.sidebar.header("Pruning Settings")
prune_level = st.sidebar.slider("Max Depth:", min_value=1, max_value=10, value=3, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split:", min_value=2, max_value=20, value=2, step=1)
ccp_alpha = st.sidebar.slider("Cost Complexity (alpha):", min_value=0.0, max_value=0.05, value=0.0, step=0.001, format="%.3f")

# Helper functions
def train_tree(max_depth=None, min_samples_split=2, ccp_alpha=0.0):
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    return clf

def get_tree_metrics(clf):
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    depth = clf.get_depth()
    nodes = clf.get_n_leaves()
    return accuracy, depth, nodes

def plot_tree_image(clf, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=target_names, ax=ax)
    ax.set_title(title)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualization", "🧮 Solved Example", "🧠 Quiz"])

with tab1:
    st.header("Learn About Tree Pruning")
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3>What is Tree Pruning?</h3>
    <p>Tree pruning is like trimming a bush in your garden. Just as you cut back branches to help the plant grow better, 
    we 'trim' parts of a decision tree to help it make better predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Do We Prune Trees?</h3>
    <ul>
        <li><span class="highlight">Prevent Overfitting:</span> Stop the tree from 'memorizing' the training data.</li>
        <li><span class="highlight">Improve Generalization:</span> Help the tree make better predictions on new data.</li>
        <li><span class="highlight">Increase Interpretability:</span> Make the tree easier for humans to understand.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>How Do We Prune?</h3>
    <p>There are several ways to prune a tree:</p>
    <ul>
        <li><span class="highlight">Max Depth:</span> Limit how deep the tree can grow.</li>
        <li><span class="highlight">Min Samples Split:</span> Only split a node if it has a minimum number of samples.</li>
        <li><span class="highlight">Cost Complexity Pruning:</span> Balance the complexity of the tree with its accuracy.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Tree Pruning in Action")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Tree")
        clf_original = train_tree()
        accuracy_original, depth_original, nodes_original = get_tree_metrics(clf_original)
        st.image(plot_tree_image(clf_original, "Original Tree"))
        st.metric("Test Accuracy", f"{accuracy_original:.2f}")
        st.metric("Tree Depth", depth_original)
        st.metric("Number of Leaves", nodes_original)
    
    with col2:
        st.subheader("Pruned Tree")
        clf_pruned = train_tree(max_depth=prune_level, min_samples_split=min_samples_split, ccp_alpha=ccp_alpha)
        accuracy_pruned, depth_pruned, nodes_pruned = get_tree_metrics(clf_pruned)
        st.image(plot_tree_image(clf_pruned, f"Pruned Tree (Max Depth: {prune_level}, Min Samples Split: {min_samples_split}, CCP Alpha: {ccp_alpha:.3f})"))
        st.metric("Test Accuracy", f"{accuracy_pruned:.2f}")
        st.metric("Tree Depth", depth_pruned)
        st.metric("Number of Leaves", nodes_pruned)

with tab3:
    st.header("Solved Example: Impact of Pruning")
    
    prune_levels = range(1, 11)
    accuracies = []
    depths = []
    nodes = []
    
    for level in prune_levels:
        clf = train_tree(max_depth=level, min_samples_split=min_samples_split, ccp_alpha=ccp_alpha)
        accuracy, depth, node_count = get_tree_metrics(clf)
        accuracies.append(accuracy)
        depths.append(depth)
        nodes.append(node_count)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(prune_levels), y=accuracies, mode='lines+markers', name='Accuracy'))
    fig.update_layout(title='Impact of Max Depth on Test Accuracy',
                      xaxis_title='Max Tree Depth',
                      yaxis_title='Test Accuracy')
    st.plotly_chart(fig, use_container_width=True)
    
    optimal_depth = np.argmax(accuracies) + 1
    st.write(f"The optimal depth for this dataset appears to be around {optimal_depth}, with a test accuracy of {max(accuracies):.2f}.")
    
    # Additional visualizations
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=list(prune_levels), y=depths, name='Tree Depth'))
    fig_metrics.add_trace(go.Bar(x=list(prune_levels), y=nodes, name='Number of Leaves'))
    fig_metrics.update_layout(title='Tree Complexity vs Max Depth',
                              xaxis_title='Max Tree Depth',
                              yaxis_title='Count',
                              barmode='group')
    st.plotly_chart(fig_metrics, use_container_width=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What happens to the number of leaves as we increase the maximum depth of the tree?",
            "options": ["It always decreases", "It always increases", "It stays the same", "It first increases then may plateau"],
            "correct": 3,
            "explanation": "As we increase the maximum depth, the number of leaves generally increases, but it may plateau once the tree has fully split the data or reached its maximum possible depth for the given dataset."
        },
        {
            "question": "What is the main goal of tree pruning?",
            "options": ["To make the tree bigger", "To make the tree smaller", "To increase accuracy on training data", "To make predictions faster"],
            "correct": 1,
            "explanation": "The main goal of tree pruning is to make the tree smaller. This helps prevent overfitting and can improve the model's performance on new, unseen data."
        },
        {
            "question": "In our example, what typically happens to the accuracy as we increase the maximum depth?",
            "options": ["It only increases", "It only decreases", "It increases then may plateau or slightly decrease", "It stays exactly the same"],
            "correct": 2,
            "explanation": "Typically, as we increase the maximum depth, the accuracy increases up to a certain point, then it may plateau or slightly decrease. This pattern shows the trade-off between model complexity and generalization."
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
st.sidebar.info("This app demonstrates the effect of tree pruning on decision tree models. Adjust the pruning parameters and explore the different tabs to learn more!")
