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
    if dataset_name == "Boston Housing":
        data = datasets.load_boston()
    elif dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    elif dataset_name == "California Housing":
        data = datasets.fetch_california_housing()
    X = data.data
    y = data.target
    return X, y, data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def main():
    st.title("AutoML Regression App")
    
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Boston Housing", "Diabetes", "California Housing"))
    
    X, y, data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Dataset")
    st.write(data['DESCR'])
    st.write("Dataset shape:", X.shape)
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
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
    
    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        mse, mae, r2 = evaluate_model(model, X_test, y_test)
        results.append([model_name, mse, mae, r2])
    
    df_results = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
    df_results = df_results.sort_values(by=["MSE", "MAE"], ascending=True)
    
    st.subheader("Evaluation Results")
    st.table(df_results)
    
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [model for model_name, model in models if model_name == best_model_name][0]
    
    st.subheader("Best Model")
    st.write(f"The best-performing model is: {best_model_name}")
    
if __name__ == "__main__":
    main()