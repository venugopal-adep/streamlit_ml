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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualize", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("Understanding Cross-Validation")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Cross-Validation?</h3>
    <p>Cross-validation is like a practice test for your machine learning model. Imagine you're studying for an exam:</p>
    <ul>
        <li>You have a set of practice questions (your data).</li>
        <li>You want to know how well you'll do on the real test (unseen data).</li>
        <li>Cross-validation helps you estimate this by cleverly using your practice questions.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Types of Cross-Validation</h3>
    <h4>1. K-Fold Cross-Validation</h4>
    <p>Think of K-Fold CV as dividing your study material into K equal parts:</p>
    <ul>
        <li>You study K-1 parts and test yourself on the remaining part.</li>
        <li>You repeat this K times, each time testing on a different part.</li>
        <li>Your final score is the average of all K tests.</li>
    </ul>
    <h4>2. Leave-One-Out Cross-Validation (LOOCV)</h4>
    <p>LOOCV is like having a personal tutor who tests you on each question individually:</p>
    <ul>
        <li>You study all questions except one.</li>
        <li>You test yourself on that single question.</li>
        <li>You repeat this for every question.</li>
        <li>Your final score is the average of all these mini-tests.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Cross-Validation?</h3>
    <ul>
        <li><span class="highlight">Reliable Performance Estimation:</span> It gives you a more accurate idea of how well your model will perform on new, unseen data.</li>
        <li><span class="highlight">Overfitting Detection:</span> It helps you spot if your model is memorizing the data instead of learning general patterns.</li>
        <li><span class="highlight">Model Selection:</span> It aids in choosing the best model among different options.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Cross-Validation in Action")
    
    X, y = st.session_state['X'], st.session_state['y']
    
    if cv_type == 'K-Fold':
        mse_scores = perform_k_fold_cv(X, y, n_splits=n_splits)
    else:
        mse_scores = perform_loocv(X, y)
    
    # Visualizing the MSE scores
    fig = px.bar(x=[f'Fold {i+1}' for i in range(len(mse_scores))], y=mse_scores, labels={'x': '', 'y': 'MSE'},
                 title=f"Mean Squared Error Across Each {cv_type}")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(marker_color='#8a2be2')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean MSE", f"{np.mean(mse_scores):.4f}", delta=f"{np.mean(mse_scores)-np.median(mse_scores):.4f}")
    with col2:
        st.metric("Std Dev of MSE", f"{np.std(mse_scores):.4f}")
    with col3:
        st.metric("Number of Folds", len(mse_scores))
    
    st.markdown("""
    <div style="background-color: #fffacd; padding: 10px; border-radius: 5px;">
    <p><strong>Interpretation:</strong> Lower MSE values indicate better model performance. The standard deviation shows how consistent the model's performance is across different folds.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Solved Example: Comparing K-Fold and LOOCV")
    
    X_example, y_example = generate_data(n_samples=50, seed=42)
    
    k_fold_scores = perform_k_fold_cv(X_example, y_example, n_splits=5)
    loocv_scores = perform_loocv(X_example, y_example)
    
    st.write("Let's compare 5-Fold CV and LOOCV on a small dataset (50 samples).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("5-Fold CV Mean MSE", f"{np.mean(k_fold_scores):.4f}")
        st.metric("5-Fold CV Std Dev of MSE", f"{np.std(k_fold_scores):.4f}")
    with col2:
        st.metric("LOOCV Mean MSE", f"{np.mean(loocv_scores):.4f}")
        st.metric("LOOCV Std Dev of MSE", f"{np.std(loocv_scores):.4f}")
    
    st.markdown("""
    <div style="background-color: #e0ffff; padding: 15px; border-radius: 8px; margin-top: 20px;">
    <h4>Observations:</h4>
    <ul>
        <li>LOOCV typically provides a lower variance estimate of the model's performance.</li>
        <li>K-Fold CV might give a more realistic estimate of performance on new data.</li>
        <li>LOOCV can be computationally expensive for larger datasets.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of cross-validation?",
            "options": ["To increase model complexity", "To assess model performance on unseen data", "To speed up model training", "To reduce the need for test data"],
            "correct": 1,
            "explanation": "Cross-validation helps us estimate how well our model will perform on new, unseen data. It's like a practice test for our machine learning model."
        },
        {
            "question": "In K-Fold cross-validation, what does K represent?",
            "options": ["The number of features", "The number of samples", "The number of splits in the data", "The number of iterations"],
            "correct": 2,
            "explanation": "K represents the number of groups (folds) that we split our data into. If K=5, we divide our data into 5 parts, using 4 for training and 1 for testing, repeating this process 5 times."
        },
        {
            "question": "What is a potential drawback of Leave-One-Out Cross-Validation (LOOCV)?",
            "options": ["It's not accurate", "It can be computationally expensive", "It always leads to overfitting", "It can't be used with large datasets"],
            "correct": 1,
            "explanation": "While LOOCV can provide a detailed assessment, it requires fitting the model as many times as there are samples in the dataset. This can be very time-consuming for large datasets."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Incorrect. Let's review this concept.")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates different cross-validation techniques. Adjust the settings and explore the different tabs to learn more!")
