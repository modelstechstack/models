import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Start a new MLflow run
with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(model, "model")

    # Save the model to a .pkl file
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    mlflow.log_artifact(model_filename)  # This logs the .pkl file to the artifact store - testing v1.0
