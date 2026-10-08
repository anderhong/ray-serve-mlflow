from ray import serve
from fastapi import FastAPI
import ray

# Initialize Ray.
ray.init()

# Create a FastAPI application.
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class HelloWorld:
    @app.get("/")
    async def say_hello(self):
        return {"message": "Hello from Ray Serve! You're doing great!"}

# Bind the deployment.
hello_app = HelloWorld.bind()
