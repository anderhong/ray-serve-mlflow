from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def trace_model(model_name, version):
    # 1. 拎 Model Version 嘅資訊
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id
    
    print(f"📦 Model: {model_name} (Version {version})")
    print(f"🔗 Source Run ID: {run_id}")
    
    # 2. 用 Run ID 去拎返完整嘅 Run 資訊
    run = client.get_run(run_id)
    
    print("\n📊 Parameters:")
    for key, value in run.data.params.items():
        print(f"  - {key}: {value}")
    
    print("\n📈 Metrics:")
    for key, value in run.data.metrics.items():
        print(f"  - {key}: {value}")
    
    # 3. 嘗試拎 Git Commit (如果有)
    tags = run.data.tags
    if "mlflow.source.git.commit" in tags:
        print(f"\n🔖 Git Commit: {tags['mlflow.source.git.commit']}")
    else:
        print("\n🔖 Git Commit: Not available (Git not configured)")

if __name__ == "__main__":
    # 測試：Trace Version 1 同 Version 2
    trace_model("ForecastModel", 1)
    print("\n" + "="*50 + "\n")
    trace_model("ForecastModel", 2)