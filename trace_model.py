from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def trace_model(model_name, version):
    # 1. Retrieve model-version information.
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id
    
    print(f"📦 Model: {model_name} (Version {version})")
    print(f"🔗 Source Run ID: {run_id}")
    
    # 2. Retrieve the complete run information using the run ID.
    run = client.get_run(run_id)
    
    print("\n📊 Parameters:")
    for key, value in run.data.params.items():
        print(f"  - {key}: {value}")
    
    print("\n📈 Metrics:")
    for key, value in run.data.metrics.items():
        print(f"  - {key}: {value}")
    
    # 3. Retrieve the Git commit, if available.
    tags = run.data.tags
    if "mlflow.source.git.commit" in tags:
        print(f"\n🔖 Git Commit: {tags['mlflow.source.git.commit']}")
    else:
        print("\n🔖 Git Commit: Not available (Git not configured)")

if __name__ == "__main__":
    # Test by tracing versions 1 and 2.
    trace_model("ForecastModel", 1)
    print("\n" + "="*50 + "\n")
    trace_model("ForecastModel", 2)
