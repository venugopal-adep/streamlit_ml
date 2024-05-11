import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set page title
st.set_page_config(page_title="Bootstrap Aggregation (Bagging) Demo")

# Title and description
st.write("## Bootstrap Aggregation (Bagging)")
st.write("**Developed by : Venugopal Adep**")
st.write("This interactive application demonstrates the concept of Bootstrap Aggregation (Bagging), a technique of merging the outputs of various models to get a final result.")

# Generate sample data
X, y = make_classification(n_samples=200, n_classes=2, random_state=42)

# Create a scatter plot of the original data
fig_original = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='Viridis'))])
fig_original.update_layout(title='Original Data', xaxis_title='Feature 1', yaxis_title='Feature 2')
st.plotly_chart(fig_original)

# User input for number of classifiers and sample size using sidebar
n_classifiers = st.sidebar.slider("Number of classifiers:", min_value=1, max_value=10, value=3, step=1)
sample_size = st.sidebar.slider("Sample size (% of original data):", min_value=10, max_value=100, value=50, step=10)

# Bootstrap sampling and training individual classifiers
classifiers = []
for i in range(n_classifiers):
    indices = np.random.choice(len(X), size=int(len(X) * sample_size / 100), replace=True)
    X_subset, y_subset = X[indices], y[indices]
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_subset, y_subset)
    classifiers.append(clf)

    # Create a scatter plot for each classifier
    fig_classifier = go.Figure(data=[go.Scatter(x=X_subset[:, 0], y=X_subset[:, 1], mode='markers', marker=dict(color=y_subset, colorscale='Viridis'))])
    fig_classifier.update_layout(title=f'Classifier {i+1}', xaxis_title='Feature 1', yaxis_title='Feature 2')
    st.plotly_chart(fig_classifier)

# Aggregate the predictions of individual classifiers
bagging_clf = BaggingClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=n_classifiers)
bagging_clf.fit(X, y)

# Make predictions using the bagging classifier
y_pred = bagging_clf.predict(X)

# Create a scatter plot of the bagging classifier's predictions
fig_bagging = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y_pred, colorscale='Viridis'))])
fig_bagging.update_layout(title='Bagging Classifier Predictions', xaxis_title='Feature 1', yaxis_title='Feature 2')
st.plotly_chart(fig_bagging)

# Evaluate the bagging classifier
accuracy = accuracy_score(y, y_pred)
st.write(f"Bagging Classifier Accuracy: {accuracy:.2f}")

# Explanation
st.subheader("Explanation")
st.write("Bootstrap Aggregation (Bagging) is a technique that combines the outputs of multiple models to obtain a final result. It reduces the chances of overfitting by training each model only with a randomly chosen subset of the training data.")
st.write("In this application, you can adjust the number of classifiers and the sample size used for training each classifier. The individual classifiers are trained on different subsets of the data, and their predictions are aggregated to make the final predictions.")
st.write("The scatter plots show the original data, the subsets used for training each classifier, and the final predictions made by the bagging classifier. The accuracy of the bagging classifier is also displayed.")
