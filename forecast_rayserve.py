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

    # 注意：呢度嘅縮排正確，屬於 Class 入面
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.2)
    async def predict_batch(self, requests):
        # 如果冇 Request 就早啲返返去
        if not requests:
            return []
        
        # 確保每一個 Request 都有 temperature 呢個 attribute
        temperatures = []
        for req in requests:
            if hasattr(req, 'temperature'):
                temperatures.append(req.temperature)
            else:
                # 如果冇 temperature，用一個預設值
                temperatures.append(20.0)
        
        # 轉做 Numpy Array 做 Prediction
        temperatures_np = np.array(temperatures).reshape(-1, 1)
        predictions = self.model.predict(temperatures_np)
        return predictions.tolist()

    @app.post("/forecast")
    async def predict(self, request: ForecastRequest):
        # 直接將 request 放入 batch queue
        batch_result = await self.predict_batch(request)
        
        # 安全檢查：確保 batch_result 係一個 list 而且有內容
        if isinstance(batch_result, list) and len(batch_result) > 0:
            return {"prediction": batch_result[0]}
        else:
            # Fallback 機制：如果 batch 回傳嘅格式有問題，直接用 model 做 prediction
            features = np.array([[request.temperature]])
            prediction = self.model.predict(features)
            return {"prediction": prediction[0], "note": "fallback"}

# 呢行一定要喺 Class 嘅外面，同埋係最底
forecast_app = ForecastModel.bind()