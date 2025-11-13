from typing import Union
from fastapi import FastAPI, Query, Request
import pickle
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, REGISTRY
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


# --------------------------
# Constants 版本信息
# --------------------------
APP_VERSION = "0.0"
MODEL_DIR = "models"  # 存放不同版本模型的目录
VERSION = "1.0"  # 默认最新模型版本
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"model_v{VERSION}.pkl")


# ======================
# Load model function
# ======================
def load_model(version: str):
    model_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"Model version {version} not found at {model_path}")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

# 默认加载最新模型
model_data = load_model(VERSION)
w = model_data["w"]
b = model_data["b"]
x_scaler = model_data["x_scaler"]
features = model_data["features"]

# ======================
# Initialize FastAPI
# ======================
app = FastAPI(title="Boston Housing API", version=APP_VERSION)

# create a new Gauge to record prediction value
PREDICTION_VALUE = Gauge("PREDICTION_VALUE", "Model prediction output value", registry=REGISTRY)

# initialize Prometheus monitoring
Instrumentator().instrument(app).expose(app)

# ======================
# Define input schema
# ======================
class HouseData(BaseModel):
    CRIM: float = Field(..., ge=0)
    ZN: float = Field(..., ge=0)
    INDUS: float
    CHAS: int = Field(..., ge=0, le=1)
    NOX: float = Field(..., ge=0, le=1)
    RM: float = Field(..., ge=0)
    AGE: float = Field(..., ge=0, le=100)
    DIS: float = Field(..., ge=0)
    RAD: int = Field(..., ge=1)
    TAX: float = Field(..., ge=0)
    PTRATIO: float = Field(..., ge=0)
    B: float = Field(..., ge=0)
    LSTAT: float = Field(..., ge=0)

# ======================
# Predict endpoint
# ======================

@app.post("/predict")
def predict_endpoint(
    input: HouseData,
    model_version: str = Query(None, description="Optional model version")
):
    version_to_use = model_version if model_version else VERSION

    # 加载指定版本模型
    model_data = load_model(version_to_use)
    w = model_data["w"]
    b = model_data["b"]
    x_scaler = model_data["x_scaler"]
    features = model_data["features"]

    # 构造输入特征
    X = np.array([[getattr(input, f) for f in features]])
    X_scaled = x_scaler.transform(X)

    # 预测
    y_pred = float(np.dot(X_scaled, w) + b)

    PREDICTION_VALUE.set(y_pred)

    # metadata
    metadata = {
        "app_version": APP_VERSION,
        "model_version": version_to_use,
        "prediction_time": datetime.utcnow().isoformat() + "Z"
    }

    # 添加 OpenTelemetry trace 信息
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute("prediction.value", y_pred)
        current_span.set_attribute("prediction.model_version", version_to_use)
        current_span.set_attribute("prediction.app_version", APP_VERSION)
        current_span.set_attribute("prediction.features", str(features))
        current_span.set_attribute("prediction.input", str(X.tolist()))
        current_span.set_attribute("prediction.timestamp", metadata["prediction_time"])

    return {"prediction": y_pred, "metadata": metadata}

# ======================
# Other routes
# ======================
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/foobar")
async def foobar():
    return {"message": "hello world"}

#  初始化 TracerProvider
trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({"service.name": "fastapi-service"})
    )
)

# 选择导出方式（OTLP/Jaeger）
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")

# Jaeger 示例（如果 docker-compose 里跑了 Jaeger）：
# jaeger_exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# 启用 FastAPI 自动采集
FastAPIInstrumentor.instrument_app(app)
#  启用 requests 库的采集（如果项目中发 HTTP 请求）
RequestsInstrumentor().instrument()


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# ======================
# Example Input:
# ======================
# {
#   "CRIM": 0.03,
#   "ZN": 18.0,
#   "INDUS": 2.3,
#   "CHAS": 0,
#   "NOX": 0.4,
#   "RM": 6.5,
#   "AGE": 45.0,
#   "DIS": 5.2,
#   "RAD": 1,
#   "TAX": 290.0,
#   "PTRATIO": 17.8,
#   "B": 390.0,
#   "LSTAT": 12.5
# }
