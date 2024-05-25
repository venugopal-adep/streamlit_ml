import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pandas as pd

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

def main():
    st.title("AutoML Regression App")
    st.write('**Developed by : Venugopal Adep**')
    
    dataset_options = [
        "Diabetes",
        "Linnerud",
        "Breast Cancer"
    ]
    dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    X, y, target_name, data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Dataset")
    if isinstance(X, pd.DataFrame):
        st.write("First 10 rows of the dataset:")
        st.write(X.head(10))
    else:
        df = pd.DataFrame(data=X, columns=data.feature_names)
        df[target_name] = y
        st.write("First 10 rows of the dataset:")
        st.write(df.head(10))
    
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
    st.write("**Target variable:**", target_name)
    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        mse, mae, r2 = evaluate_model(model, X_test, y_test)
        results.append([model_name, mse, mae, r2])
    
    df_results = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
    
    sort_by = st.selectbox("Sort results by", ["MSE", "MAE", "R2"])
    ascending = True if sort_by in ["MSE", "MAE"] else False
    df_results = df_results.sort_values(by=sort_by, ascending=ascending)
    
    st.subheader("Evaluation Results")
    st.table(df_results)
    
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [model for model_name, model in models if model_name == best_model_name][0]
    
    st.subheader("Best Model")
    st.write(f"The best-performing model based on {sort_by} is: {best_model_name}")
    
    st.subheader("Dataset Description")
    if hasattr(data, 'DESCR'):
        st.write(data.DESCR)
    st.write("Dataset shape:", X.shape)
    
    
if __name__ == "__main__":
    main()
