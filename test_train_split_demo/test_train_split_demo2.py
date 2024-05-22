import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

# Load datasets
datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer()
}

# Initialize the app
st.title('Train-Test Split Interactive Demo')
st.write('**Developed by : Venugopal Adep**')
st.write("""
This interactive demo allows you to visualize the train-test split and understand its impact on a machine learning model.
""")

# Sidebar options
dataset_name = st.sidebar.selectbox('Select Dataset', list(datasets.keys()))
test_size = st.sidebar.slider('Test Set Size (%)', min_value=10, max_value=50, value=25, step=5) / 100.0

# Place Resample button in the sidebar
if st.sidebar.button('Resample'):
    resample_data = True
else:
    resample_data = False

# Load selected dataset
data = datasets[dataset_name]
X, y = data['data'], data['target']
feature_names = data['feature_names']
class_names = data['target_names']

# Split the data
if resample_data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train a simple K-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

# Display model metrics
st.write(f"Model Accuracy: **{accuracy:.2f}**")
st.write(f"Model Recall: **{recall:.2f}**")
st.write(f"Model Precision: **{precision:.2f}**")

# Plotting
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['type'] = 'Train'
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test['type'] = 'Test'
df_combined = pd.concat([df_train, df_test])

colors = {'Train': 'red', 'Test': 'green'}  # Assigning colors
fig = px.scatter_3d(df_combined, x=feature_names[0], y=feature_names[1], z=feature_names[2], color='type', 
                    color_discrete_map=colors, title='Train-Test Split 3D Visualization')
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=600)  # Adjust layout and size
st.plotly_chart(fig)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

fig_cm = px.imshow(cm_df, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"), color_continuous_scale=px.colors.sequential.Viridis)
fig_cm.update_layout(title="Confusion Matrix")
st.plotly_chart(fig_cm)
