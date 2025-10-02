from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
from ..models.state_models import SupervisorState
from ..services.orchestrator import run_sync
from fastapi.encoders import jsonable_encoder
import asyncio
import ast 
router = APIRouter()
from datetime import datetime
def serialize_state(obj):
    """Recursively convert objects into JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: serialize_state(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_state(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "model_dump"):
        return serialize_state(obj.model_dump())
    else:
        return obj 
    

@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = None

    try:
        while True:
            msg = await websocket.receive_text()
            msg = ast.literal_eval(msg)
            
            print("Received message:", type(msg))

            msg = msg.get("message", "")
            print("Received message:", msg)

            if state is None:
                state = SupervisorState(user_query=msg)
            else:
                state.context[f"user_message_{len(state.context)+1}"] = msg
                state.current_task = msg

            # run_sync returns an async generator; iterate through it to get the final state.
            final_state = None
            try:
                async for chunk in run_sync(state):
                    await websocket.send_json(serialize_state(chunk))
                    final_state= chunk
            except asyncio.CancelledError:
                break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                break
            # if isinstance(final_state, dict):
            #     state = SupervisorState.model_validate(final_state)
            await websocket.send_json({
                "type": "final",
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
# ok health check


@router.post("/chat")
async def chat_endpoint(payload: dict):
    
    state = None
    print("payload:", payload)
    try:
        while True:
            msg = payload.get("message", "")

            if state is None:
                state = SupervisorState(user_query=msg)
            else:
                state.context[f"user_message_{len(state.context)+1}"] = msg
                state.current_task = msg

            # run_sync returns an async generator; iterate through it to get the final state.
            final_state = None
            st= await run_sync(state)
            final_state =  st

            if isinstance(final_state, dict):
                if "supervisor" in final_state:
                    state = SupervisorState.model_validate(final_state["supervisor"])
                else:
                    state = SupervisorState.model_validate(final_state)
            else:
                state = final_state

            # Serialize the entire state before sending
            serialized_state = serialize_state(state)
            return {
                "state": serialized_state,
            }

    except WebSocketDisconnect:
        print("WebSocket disconnected")




@router.get("/health")
async def health_check():
    return {"status": "ok"}