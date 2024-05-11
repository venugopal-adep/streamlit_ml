import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(n_samples=100, n_features=1, noise=15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    return X.reshape(-1, 1), y

def perform_k_fold_cv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    mse_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    return mse_scores

def perform_loocv(X, y):
    loo = LeaveOneOut()
    mse_scores = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    return mse_scores

# Streamlit UI setup
st.title("Cross-Validation Techniques Demonstration")

# Sidebar for user input and data regeneration
st.sidebar.header("Configuration and Data")
cv_type = st.sidebar.radio("Select Cross-Validation Type", ('K-Fold', 'LOOCV'))
if cv_type == 'K-Fold':
    n_splits = st.sidebar.slider("Number of Folds", 2, 10, 5)
else:
    n_splits = 1

if st.sidebar.button('Regenerate Data'):
    new_seed = np.random.randint(10000)
    X, y = generate_data(seed=new_seed)
    st.session_state['X'], st.session_state['y'] = X, y
else:
    if 'X' not in st.session_state or 'y' not in st.session_state:
        X, y = generate_data()
        st.session_state['X'], st.session_state['y'] = X, y
    else:
        X, y = st.session_state['X'], st.session_state['y']

if cv_type == 'K-Fold':
    mse_scores = perform_k_fold_cv(X, y, n_splits=n_splits)
else:
    mse_scores = perform_loocv(X, y)

# Visualizing the MSE scores
fig = px.bar(x=[f'Fold {i+1}' for i in range(len(mse_scores))], y=mse_scores, labels={'x': '', 'y': 'MSE'},
             title=f"Mean Squared Error Across Each {cv_type}")
st.plotly_chart(fig)

# Tool usage instructions and explanations
st.subheader("How to Use This Tool")
st.write("""
Select the cross-validation method using the sidebar. Adjust the number of folds for K-Fold cross-validation. The chart displays the Mean Squared Error (MSE) for each fold or iteration to assess model performance. Use the 'Regenerate Data' button to create new datasets for further experimentation.
""")

st.subheader("Understanding Cross-Validation")
st.write("""
**K-Fold Cross-Validation** involves splitting the dataset into K consecutive folds, then using each fold once as the validation while the remaining K-1 folds form the training set. This method is widely used to evaluate the performance of predictive models.

**Leave-One-Out Cross-Validation (LOOCV)** involves using each instance of the dataset as a single test instance, and the rest as the training set. It is particularly useful when the dataset is small.

**Example**: In K-Fold with K=5 and 100 observations, each fold consists of 20 observations. Each fold is tested using the rest as training data. LOOCV would test one observation at a time against 99 others.
""")

st.write("### Insights")
st.write("This visualization shows the distribution of Mean Squared Errors for each fold in the selected cross-validation method, providing insights into model performance consistency and potential overfitting issues.")
