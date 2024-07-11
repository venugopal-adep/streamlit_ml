import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(page_title="Cross-Validation Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🔀 Cross-Validation Techniques Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of cross-validation in assessing model performance!")

# Helper functions
def generate_data(n_samples=100, n_features=1, noise=15, seed=42):
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

# Sidebar
st.sidebar.header("Configuration")
cv_type = st.sidebar.radio("Select Cross-Validation Type", ('K-Fold', 'LOOCV'))
if cv_type == 'K-Fold':
    n_splits = st.sidebar.slider("Number of Folds", 2, 10, 5)
else:
    n_splits = 1

# Generate data once
if 'X' not in st.session_state or 'y' not in st.session_state:
    X, y = generate_data()
    st.session_state['X'], st.session_state['y'] = X, y

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Cross-Validation in Action")
    
    X, y = st.session_state['X'], st.session_state['y']
    
    if cv_type == 'K-Fold':
        mse_scores = perform_k_fold_cv(X, y, n_splits=n_splits)
    else:
        mse_scores = perform_loocv(X, y)
    
    # Visualizing the MSE scores
    fig = px.bar(x=[f'Fold {i+1}' for i in range(len(mse_scores))], y=mse_scores, labels={'x': '', 'y': 'MSE'},
                 title=f"Mean Squared Error Across Each {cv_type}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean MSE", f"{np.mean(mse_scores):.4f}")
    with col2:
        st.metric("Std Dev of MSE", f"{np.std(mse_scores):.4f}")
    with col3:
        st.metric("Number of Folds", len(mse_scores))

with tab2:
    st.header("Solved Example: Comparing K-Fold and LOOCV")
    
    X_example, y_example = generate_data(n_samples=50, seed=42)
    
    k_fold_scores = perform_k_fold_cv(X_example, y_example, n_splits=5)
    loocv_scores = perform_loocv(X_example, y_example)
    
    st.write("We'll compare 5-Fold CV and LOOCV on a small dataset (50 samples).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("5-Fold CV Mean MSE", f"{np.mean(k_fold_scores):.4f}")
        st.metric("5-Fold CV Std Dev of MSE", f"{np.std(k_fold_scores):.4f}")
    with col2:
        st.metric("LOOCV Mean MSE", f"{np.mean(loocv_scores):.4f}")
        st.metric("LOOCV Std Dev of MSE", f"{np.std(loocv_scores):.4f}")
    
    st.write("As we can see, LOOCV typically provides a lower variance estimate of the model's performance, but it can be computationally expensive for larger datasets.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of cross-validation?",
            "options": ["To increase model complexity", "To assess model performance on unseen data", "To speed up model training", "To reduce the need for test data"],
            "correct": 1,
            "explanation": "Cross-validation is primarily used to assess how well a model will generalize to an independent dataset. It helps in understanding the model's performance on unseen data and detecting issues like overfitting."
        },
        {
            "question": "In K-Fold cross-validation, what does K represent?",
            "options": ["The number of features", "The number of samples", "The number of splits in the data", "The number of iterations"],
            "correct": 2,
            "explanation": "In K-Fold cross-validation, K represents the number of groups that a given data sample is to be split into. For example, if K=5, the data is split into 5 folds, each used once as a validation set while the other K-1 folds form the training set."
        },
        {
            "question": "What is a potential drawback of Leave-One-Out Cross-Validation (LOOCV)?",
            "options": ["It's not accurate", "It can be computationally expensive", "It always leads to overfitting", "It can't be used with large datasets"],
            "correct": 1,
            "explanation": "While LOOCV provides a nearly unbiased estimate of the model's performance, it can be computationally expensive, especially for large datasets. This is because it requires fitting the model n times, where n is the number of samples in the dataset."
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
    st.header("Learn More About Cross-Validation")
    st.markdown("""
    Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. It's primarily used in applied machine learning to estimate the skill of a model on unseen data.

    Key benefits of Cross-Validation:
    1. **Model Evaluation**: Helps in assessing how the results of a statistical analysis will generalize to an independent data set.
    2. **Overfitting Detection**: Helps in detecting overfitting, i.e., when a model performs well on training data but poorly on unseen data.
    3. **Model Selection**: Aids in selecting the best model among different options.

    Common Cross-Validation Techniques:
    1. **K-Fold CV**: The data is divided into k subsets, and the holdout method is repeated k times.
    2. **Leave-One-Out CV (LOOCV)**: A special case of k-fold CV where k equals the number of instances in the data.
    3. **Stratified K-Fold CV**: Ensures that the proportion of samples for each class is roughly the same in each fold.

    When to use Cross-Validation:
    - When you have a limited dataset and can't afford a large separate test set
    - When you want to tune hyperparameters of your model
    - When you need to select the best model among several candidates

    Remember, while cross-validation is a powerful technique, it's not a silver bullet. It's important to understand its assumptions and limitations when applying it to real-world problems.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates different cross-validation techniques. Adjust the settings and explore the different tabs to learn more!")
