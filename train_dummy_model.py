import mlflow
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run() as run:
    # 1. 訓練 Model
    X_train = np.array([10, 15, 20, 25, 30, 35]).reshape(-1, 1)
    y_train = np.array([110, 160, 210, 260, 310, 360])  # 將所有 Sales 數字加 10
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 2. Log Parameters 同 Metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("coefficient", model.coef_[0])
    mlflow.log_metric("intercept", model.intercept_)
    
    # 3. 🆕 用 MLflow 嘅標準方式去 Log 同註冊 Model
    # 呢個方法會自動幫你 Handle 路徑同 Format
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="forecast_model",
        signature=signature,
        registered_model_name="ForecastModel"
    )
    
    print(f"✅ Model trained and registered successfully!")
    print(f"📦 Model registered as 'ForecastModel' with run ID: {run.info.run_id}")