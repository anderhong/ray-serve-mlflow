from ray import serve
from fastapi import FastAPI
import ray

# 初始化 Ray
ray.init()

# 建立一個 FastAPI app
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class HelloWorld:
    @app.get("/")
    async def say_hello(self):
        return {"message": "Hello from Ray Serve! You're doing great!"}

# 將個 Deployment 綁定
hello_app = HelloWorld.bind()