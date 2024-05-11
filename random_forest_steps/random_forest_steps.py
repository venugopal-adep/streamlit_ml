import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page title
st.set_page_config(page_title="Random Forest: Steps Involved")

# Title and description
st.write("## Random Forest: Steps Involved")
st.write("**Developed by : Venugopal Adep**")
st.write("This interactive application demonstrates the steps involved in the Random Forest algorithm.")

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for number of trees and maximum depth in the sidebar
n_trees = st.sidebar.slider("Number of trees:", min_value=1, max_value=100, value=10, step=1)
max_depth = st.sidebar.slider("Maximum depth of trees:", min_value=1, max_value=10, value=5, step=1)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = rf_classifier.score(X_test, y_test)

# Display the steps involved
st.subheader("Steps Involved:")
steps = [
    "1. Selection of a random subsample of a given dataset.",
    "2. Using attribute selection indicators create a decision tree for each subsample and record the prediction outcome from each model.",
    "3. Applying the voting/averaging method over predicted outcomes of individual models.",
    "4. Considering the final results as the average value or most voted value."
]

for step in steps:
    st.write(step)

# Display the number of trees and maximum depth
st.subheader("Hyperparameters:")
st.write(f"Number of trees: {n_trees}")
st.write(f"Maximum depth of trees: {max_depth}")

# Display the accuracy
st.subheader("Accuracy:")
st.write(f"Accuracy: {accuracy:.2f}")

# Plot the feature importance
st.subheader("Feature Importance:")
feature_importance = rf_classifier.feature_importances_
fig = go.Figure(data=[go.Bar(x=[f'Feature {i+1}' for i in range(len(feature_importance))], y=feature_importance)])
fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance')
st.plotly_chart(fig)

# Plot the voting process
st.subheader("Voting Process:")
tree_predictions = [tree.predict(X_test) for tree in rf_classifier.estimators_]
vote_counts = [sum(pred) for pred in zip(*tree_predictions)]
fig = go.Figure(data=[go.Bar(x=[f'Class {i}' for i in range(2)], y=vote_counts)])
fig.update_layout(title='Voting Process', xaxis_title='Classes', yaxis_title='Vote Count')
st.plotly_chart(fig)

# Plot the final prediction
st.subheader("Final Prediction:")
fig = go.Figure(data=[go.Bar(x=['Final Prediction'], y=[accuracy], name='Accuracy')])
fig.update_layout(title='Final Prediction', xaxis_title='Result', yaxis_title='Accuracy')
st.plotly_chart(fig)
