import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.networks import router as networks_router
from .routes.weights import router as weights_router
from .routes.simulate import router as simulate_router
from .routes.train import router as train_router
from .routes.results import router as results_router
from .ws.simulate import router as ws_simulate_router
from .ws.train import router as ws_train_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SEAL Dashboard API", version="1.0.0")


@app.on_event("startup")
def startup_init_ray():
    """Initialize Ray on server startup (must be from main thread)."""
    try:
        from .training_runner import ensure_ray
        ensure_ray()
        logger.info("Ray initialized at server startup")
    except Exception as e:
        logger.warning("Ray init at startup failed (will retry on demand): %s", e)

# CORS for frontend at localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(networks_router)
app.include_router(weights_router)
app.include_router(simulate_router)
app.include_router(train_router)
app.include_router(results_router)

# WebSocket routes
app.include_router(ws_simulate_router)
app.include_router(ws_train_router)


@app.get("/")
def root():
    return {"service": "SEAL Dashboard API", "version": "1.0.0"}
