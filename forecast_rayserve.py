from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import asyncio

# Load Model
with open("dummy_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class ForecastRequest(BaseModel):
    temperature: float

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

    # This indentation is intentional because the code belongs to the class.
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.2)
    async def predict_batch(self, requests):
        # Return early when no request is supplied.
        if not requests:
            return []
        
        # Ensure that every request has a ``temperature`` attribute.
        temperatures = []
        for req in requests:
            if hasattr(req, 'temperature'):
                temperatures.append(req.temperature)
            else:
                # Use a default value when ``temperature`` is absent.
                temperatures.append(20.0)
        
        # Convert the input to a NumPy array for prediction.
        temperatures_np = np.array(temperatures).reshape(-1, 1)
        predictions = self.model.predict(temperatures_np)
        return predictions.tolist()

    @app.post("/forecast")
    async def predict(self, request: ForecastRequest):
        # Add the request directly to the batch queue.
        batch_result = await self.predict_batch(request)
        
        # Ensure that ``batch_result`` is a non-empty list.
        if isinstance(batch_result, list) and len(batch_result) > 0:
            return {"prediction": batch_result[0]}
        else:
            # Fall back to direct model prediction if the batch response format is invalid.
            features = np.array([[request.temperature]])
            prediction = self.model.predict(features)
            return {"prediction": prediction[0], "note": "fallback"}

# This line must remain outside the class definition and at the end of the file.
forecast_app = ForecastModel.bind()
