from fastapi import FastAPI
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
import asyncio
from .config import settings
# from .api.routes import router as api_router
from .api.ws_routes import router as ws_router
from .services.orchestrator import build_graph
from fastapi.middleware.cors import CORSMiddleware




@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup logic ---
    print("Building graph...")
    try:
        build_graph()
        print("Graph built at startup")
    except Exception as e:
        print(f"Error building graph: {e}")
        raise

    yield   # Application runs here

    # --- Shutdown logic ---
    print("Shutting down gracefully...")
    # Cancel any pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    # Wait for tasks to complete with timeout
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    print("Shutdown complete")

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    # app.include_router(api_router, prefix="/v1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )
    app.include_router(ws_router)  # WebSocket routes don't need prefix
    return app

app = create_app()