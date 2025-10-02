from fastapi import FastAPI
from contextlib import asynccontextmanager

from .config import settings
# from .api.routes import router as api_router
from .api.ws_routes import router as ws_router
from .services.orchestrator import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup logic ---
    build_graph()
    print("Graph built at startup")

    yield  # Application runs here

    # --- Shutdown logic (optional) ---
    print("Shutting down backend")


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    # app.include_router(api_router, prefix="/v1")
    app.include_router(ws_router)  # WebSocket routes don’t need prefix
    return app


app = create_app()
