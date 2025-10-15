from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
from ..models.state_models import SupervisorState
from ..services.orchestrator import run_sync
from ..services.persistence import get_thread_memory, update_thread_memory
from fastapi.encoders import jsonable_encoder
from typing import List, Dict
import asyncio
import ast
router = APIRouter()
from datetime import datetime

def serialize_state(obj):
    """
    Recursively serialize any object to JSON-compatible format.
    Handles Pydantic models, datetime objects, and nested structures.
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return obj.model_dump(mode='json')
    
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: serialize_state(value) for key, value in obj.items()}
    
    # Handle lists
    if isinstance(obj, list):
        return [serialize_state(item) for item in obj]
    
    # Handle tuples
    if isinstance(obj, tuple):
        return [serialize_state(item) for item in obj]
    
    # Return primitive types as-is
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # For anything else, try to convert to string
    try:
        return str(obj)
    except:
        return None


import traceback
@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            msg = await websocket.receive_text()
            msg = ast.literal_eval(msg)
            
            thread_id = msg.get("thread_id", "default")
            user_id = msg.get("user_id", "User")
            user_message = msg.get("message", "")
            print(f"Received message for thread {thread_id}: {user_message}")

            # Load existing memory for this thread
            memory = get_thread_memory(user_id, thread_id)

            # Create state with memory context
            incoming_state = SupervisorState(
                user_query=user_message,
                request_summary=memory["request_summary"] if memory else None,
                response_summary=memory["response_summary"] if memory else None
            )
            print(incoming_state)

            # Track only the final state
            final_state = None
            kwargs={
                "user_id":user_id,
            }
            try:
                async for chunk in run_sync(incoming_state, thread_id=thread_id, **kwargs):
                    chunk_data = serialize_state(chunk)

                    # Keep updating final_state (last one will have everything)
                    final_state = chunk_data

                    await websocket.send_json({
                        "type": "chunk",
                        "thread_id": thread_id,
                        "state": chunk_data
                    })

            except asyncio.CancelledError:
                print(f"WebSocket task cancelled for thread {thread_id}")
                break

            except Exception as e:
                tb = traceback.format_exc()
                print("Exception occurred:", tb)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "thread_id": thread_id,
                        "message": str(e) or tb
                    })
                except Exception:
                    # WebSocket might already be closed
                    pass
                break

            # Update memory after conversation turn
            if final_state:
                # Extract final output from the nested state structure
                state_obj = final_state.get("supervisor") if isinstance(final_state, dict) else final_state
                final_output = state_obj.get("final_output") if isinstance(state_obj, dict) else getattr(state_obj, "final_output", None)

                # Save complete conversation turn with final state (which includes full execution)
                conversation_entry = {
                    "user_query": user_message,
                    "final_state": final_state,  # Complete state with all decisions, agent_states, context
                    "final_response": final_output or "Processing...",
                }

                # Update memory with summaries and raw conversation
                update_thread_memory(
                    user_id=user_id,
                    thread_id=thread_id,
                    request_summary=f"{user_message}",
                    response_summary=f"{final_output or 'Processing...'}",
                    conversation_entry=conversation_entry
                )

            # Send final message
            try:
                await websocket.send_json({
                    "type": "final",
                    "thread_id": thread_id
                })
            except Exception:
                # WebSocket might already be closed
                pass

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except asyncio.CancelledError:
        print("WebSocket connection cancelled during shutdown")
    except Exception as e:
        print(f"Unexpected WebSocket error: {e}")

# ok health check
# Dummy user data
users = [
    {
        "id": 0,
        "name": "Alice Johnson",
        "age": 32,
        "sector_preference": "Technology",
        "risk_tolerance": "High",
        "current_investment_portfolio": 150000,
        "current_sector_investment_distribution": {
            "Technology": 50,
            "Healthcare": 30,
            "Finance": 20
        },
        "country":"UK",
        "Exchange": "LSE",
    },
    {
        "id": 1,
        "name": "Bob Smith",
        "age": 45,
        "sector_preference": "Healthcare",
        "risk_tolerance": "Medium",
        "current_investment_portfolio": 200000,
        "current_sector_investment_distribution": {
            "Technology": 20,
            "Healthcare": 50,
            "Energy": 30
        },
        "country":"UK",
        "Exchange": "LSE",
    },
    {
        "id": 2,
        "name": "Charlie Brown",
        "age": 28,
        "sector_preference": "Energy",
        "risk_tolerance": "Low",
        "current_investment_portfolio": 80000,
        "current_sector_investment_distribution": {
            "Energy": 60,
            "Finance": 25,
            "Technology": 15
        },
        "country":"UK",
        "Exchange": "LSE",

    }
]

@router.get("/users", response_model=List[Dict])
def get_users():
    return users

@router.get("/users/{user_id}", response_model=Dict)
def get_user(user_id: int):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        return user
    return {"error": "User not found"}



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


@router.get("/threads/{user_id}")
async def get_user_thread_list(user_id: str):
    """Get all thread IDs for a user with their request summaries and last updated timestamps."""
    from ..services.persistence import get_user_threads, get_thread_memory
    thread_ids = get_user_threads(user_id)

    # Build thread list with summaries and timestamps
    threads = []
    for thread_id in thread_ids:
        memory = get_thread_memory(user_id, thread_id)
        threads.append({
            "thread_id": thread_id,
            "request_summary": memory.get("request_summary", "") if memory else "",
            "last_updated": memory.get("last_updated", "") if memory else ""
        })

    # Sort by last_updated descending (most recent first)
    threads.sort(key=lambda x: x["last_updated"], reverse=True)

    return {"user_id": user_id, "threads": threads}


@router.get("/threads/{user_id}/{thread_id}")
async def get_thread_history(user_id: str, thread_id: str):
    """Get conversation history for a specific thread."""
    memory = get_thread_memory(user_id, thread_id)
    if not memory:
        return {"error": "Thread not found"}
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "memory": memory
    }


@router.delete("/threads/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    """Delete a conversation thread."""
    from ..services.persistence import clear_thread_memory
    clear_thread_memory(user_id, thread_id)
    return {"status": "deleted", "user_id": user_id, "thread_id": thread_id}