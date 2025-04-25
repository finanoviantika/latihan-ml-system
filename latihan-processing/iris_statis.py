from preprocessing import preprocess_data
from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier # Algoritma dapat disesuaikan
from sklearn.model_selection import GridSearchCV # Optimasi dapat disesuaikan

# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()

input_example = data[0:5]

#Contoh penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#cerate a new MLflow Experiment
mlflow.set_experiment("Latihan Model Statis")

with mlflow.start_run():
    mlflow.autolog()

    #parameter Grid untuk GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 505, 700],
        'max_depth': [5, 10, 15, 20, 25, 37, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #Inisiasi dan Grid Search
    # Initialize RandomForestClassifier()
    rf = RandomForestClassifier()

    #perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    #Mendapatkan parameter dan model terbaik
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    #log best parameters
    mlflow.log_params(best_params)

    #train the best model on the training set
    best_model.fit(X_train, y_train)

    #logging model
    mlflow.sklearn.log_model(
        sk_model = best_model,
        artifact_path = "model",
        input_example=input_example
    )

    #evaluate the model on the test set and log accuracy
    accuracy = best_model.score(X_test, y_test)