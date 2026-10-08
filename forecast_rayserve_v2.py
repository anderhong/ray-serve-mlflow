from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import asyncio
import mlflow
from mlflow.tracking import MlflowClient
import logging

logger = logging.getLogger("ray.serve")

# --- 1. Load a model from the MLflow Registry using an alias or the latest version. ---
def load_model_with_fallback(model_name="ForecastModel", alias="production"):
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        client = MlflowClient()
        # First try to load the model by alias.
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
            version_number = model_version.version
            logger.info(f"Found model with alias '{alias}': Version {version_number}")
        except Exception as alias_error:
            # If the alias does not exist, load the latest version.
            logger.info(f"Alias '{alias}' not found, falling back to latest version...")
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                raise Exception(f"No versions found for model {model_name}")
            # Use the latest version (the highest version number).
            latest = max(latest_versions, key=lambda v: int(v.version))
            version_number = latest.version
            logger.info(f"Loaded latest version: {version_number}")
        
        # Load Model
        model_uri = f"models:/{model_name}/{version_number}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded {model_name} Version {version_number}")
        return model
        
    except Exception as e:
        logger.info(f"Failed to load from MLflow Registry: {e}")
        logger.info("Falling back to local pickle file...")
        with open("dummy_forecast_model.pkl", "rb") as f:
            return pickle.load(f)

# Load the model, preferring the "production" alias and falling back to the latest version.
model = load_model_with_fallback(model_name="ForecastModel", alias="production")

# --- 2. FastAPI App ---
app = FastAPI()

class ForecastRequest(BaseModel):
    temperature: float

# --- 3. Ray Serve Deployment ---
@serve.deployment(
    ray_actor_options={"num_cpus": 0.5},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 5
    }
)
@serve.ingress(app)
class ForecastModel:
    def __init__(self):
        self.model = model
        logger.info("ForecastModel initialized")

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=1.0)
    async def predict_batch(self, requests):
        logger.info(f"predict_batch called with {len(requests)} requests")
        if not requests:
            logger.info("predict_batch: requests is empty, returning []")
            return []
        
        temperatures = [req.temperature for req in requests]
        logger.info(f"Temperatures: {temperatures}")
        temperatures_np = np.array(temperatures).reshape(-1, 1)
        predictions = self.model.predict(temperatures_np)
        logger.info(f"Predictions: {predictions.tolist()}")
        return predictions.tolist()

    @app.post("/forecast")
    async def predict(self, request: ForecastRequest):
        request_info = f"Received: temp={request.temperature}"
        
        try:
            batch_result = await self.predict_batch(request)
            batch_info = f"Batch result: {batch_result}, type: {type(batch_result)}"
        except Exception as e:
            batch_info = f"Batch error: {str(e)}"
            batch_result = None
        
        if isinstance(batch_result, list) and len(batch_result) > 0:
            prediction = batch_result[0]
            note = "batch"
        else:
            features = np.array([[request.temperature]])
            prediction = self.model.predict(features)[0]
            note = "fallback"
        
        return {
            "prediction": prediction,
            "note": note,
            "debug": {
                "request": request_info,
                "batch": batch_info,
                "batch_result_is_list": isinstance(batch_result, list),
                "batch_result_length": len(batch_result) if isinstance(batch_result, list) else None
            }
        }

# --- 4. Bind the deployment. ---
forecast_app = ForecastModel.bind()
