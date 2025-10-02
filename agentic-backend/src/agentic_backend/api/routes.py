from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
from ..models.state_models import SupervisorState
from ..services.orchestrator import run_sync

router = APIRouter()


@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    state = None

    try:
        while True:
            # Wait for user message
            msg = await websocket.receive_text()

            if state is None:
                # First message initializes SupervisorState
                state = SupervisorState( user_query=msg)
            else:
                # If continuing, append message to context
                state.context[f"user_message_{len(state.context)+1}"] = msg
                state.current_task = msg  # treat as new task

            # Run orchestrator synchronously (blocking)
            result_state = run_sync(state)
            state = result_state  # update working state

            # Send back trace or agent response
            await websocket.send_json({
                
                "decisions": [d.model_dump() for d in state.decisions],
                "final_output": state.final_output,
                "trace": state.get_trace(),
            })

    except WebSocketDisconnect:
        print(f"WebSocket  disconnected")



