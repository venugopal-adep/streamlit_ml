import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Set page config
st.set_page_config(page_title="AutoML Regression Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🤖 AutoML Regression Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of automated machine learning for regression tasks!")

# Helper functions
def load_dataset(dataset_name):
    if dataset_name == "Diabetes":
        data = datasets.load_diabetes()
        X = data.data
        y = data.target
        target_name = "Disease Progression"
    elif dataset_name == "Linnerud":
        data = datasets.load_linnerud()
        X = data.data
        y = data.target[:, 0]  # Select the first target variable (Physiological)
        target_name = "Physiological"
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        target_name = "Tumor Size"
    return X, y, target_name, data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

# Sidebar
st.sidebar.header("Dataset Selection")
dataset_options = ["Diabetes", "Linnerud", "Breast Cancer"]
dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)

# Load data
X, y, target_name, data = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧮 Model Evaluation", "🏆 Best Model", "📚 Dataset Description"])

with tab1:
    st.header("Data Overview")
    st.write("Dataset shape:", X.shape)
    st.write("**Target variable:**", target_name)
    
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        df[target_name] = y
    else:
        df = pd.DataFrame(data=X, columns=data.feature_names)
        df[target_name] = y
    
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))

    # Feature correlation heatmap
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, colorscale='Viridis'))
    fig.update_layout(title='Feature Correlation Heatmap', width=800, height=600)
    st.plotly_chart(fig)

with tab2:
    st.header("Model Evaluation")

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge()),
        ("Lasso Regression", Lasso()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("Gradient Boosting", GradientBoostingRegressor()),
        ("AdaBoost", AdaBoostRegressor()),
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("Support Vector Regression", SVR())
    ]

    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        mse, mae, r2 = evaluate_model(model, X_test, y_test)
        results.append([model_name, mse, mae, r2])

    df_results = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
    
    sort_by = st.selectbox("Sort results by", ["MSE", "MAE", "R2"])
    ascending = True if sort_by in ["MSE", "MAE"] else False
    df_results = df_results.sort_values(by=sort_by, ascending=ascending)

    st.table(df_results)

    # Performance comparison plot
    fig = go.Figure()
    for metric in ["MSE", "MAE", "R2"]:
        fig.add_trace(go.Bar(x=df_results["Model"], y=df_results[metric], name=metric))
    fig.update_layout(title='Model Performance Comparison', xaxis_title='Models', yaxis_title='Score', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Best Model")
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [model for model_name, model in models if model_name == best_model_name][0]

    st.write(f"The best-performing model based on {sort_by} is: **{best_model_name}**")
    st.write(f"MSE: {df_results.iloc[0]['MSE']:.4f}")
    st.write(f"MAE: {df_results.iloc[0]['MAE']:.4f}")
    st.write(f"R2: {df_results.iloc[0]['R2']:.4f}")

    # Feature importance for the best model (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': data.feature_names, 'importance': best_model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        fig = go.Figure(go.Bar(x=feature_importance['feature'], y=feature_importance['importance']))
        fig.update_layout(title=f'Feature Importance for {best_model_name}', xaxis_title='Features', yaxis_title='Importance')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Dataset Description")
    if hasattr(data, 'DESCR'):
        st.write(data.DESCR)
    st.write("Dataset shape:", X.shape)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates automated machine learning for regression tasks. Select a dataset and explore the different tabs to learn more!")
