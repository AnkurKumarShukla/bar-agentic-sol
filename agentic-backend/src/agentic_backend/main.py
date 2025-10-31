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

from fastapi import FastAPI
import asyncio
import websockets
import json



#=====================working======================

@app.post("/api/chat")
async def chat_via_rest(payload: dict):
    import asyncio, json, websockets

    message = payload.get("message", "")
    user_id = payload.get("user_id", "")
    thread_id = payload.get("thread_id", "")

    if not message:
        return {"response": "Missing 'message' field"}
    if not user_id:
        return {"response": "Missing 'user_id' field"}
    if not thread_id:
        return {"response": "Missing 'thread_id' field"}

    ws_url = "ws://127.0.0.1:8000/ws/chat"
    final_response = ""

    try:
        async with websockets.connect(ws_url) as websocket:
            send_payload = {
            "message": message,
            "user_id": user_id,
            "thread_id": thread_id
        }
            await websocket.send(json.dumps(send_payload))
            # await websocket.send(json.dumps({"message": message}))

            try:
                async for data in websocket:
                    msg = json.loads(data)
                    
                    # Check if this is the chunk message with the actual response
                    if msg.get("type") == "chunk":
                        # Extract the final_output from the nested structure
                        if "state" in msg and "supervisor" in msg["state"]:
                            supervisor = msg["state"]["supervisor"]
                            if "final_output" in supervisor:
                                final_response = supervisor["final_output"]
                    
                    # Break when we receive the final event
                    if msg.get("type") == "final":
                        break
                        
            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                print(f"Error during WebSocket communication: {e}")
        
        return {"response": final_response.strip() or "No final response received"}

    except Exception as e:
        print(f"Error connecting to WebSocket: {e}")
        return {"error": str(e)}