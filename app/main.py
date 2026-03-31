"""
S.Y.L.P.H MAIN API
============================


This module defines the FastAPI application and all HTTP endpoints. It is
designed for single-user use: one person runs one server (e.sexthon run.px)
and uses it as their personal S.Y.L.P.H backend. Many people can each run
their own copy of this code on their own machine.

ENDPOINTS:
GET /                           - Returns API name and list of endpoints.
GET /health                     - Returns status of all services (for monitoring).
POST /chat.                     - General chat puce LLM, no web search. Uses learning date
POST /chat/realtime             - Realtime chat: runs a Tavily web search first, then
                                 sends results + context to Groq. Same session as /chat.
GET /chat/history/{id}          - Returns all messages for a session (general + realtime).

SESSION:
    Both /chat and /chat/realtime use the same session_id. If you omit session_id,
    the server generates a UUID and returns it; send it back on the next request
    to continue the conversation. Sessions are saved to disk and survive restarts.
STARTUP:
    On startup, the lifespan function builds the vector store from learning_data/*.txt
    and chats_data/*.json, then creates Groq, Realtime, and Chat services. On shutdown,
    it saves all in-memory sessions to disk.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from fastapi import Request
from app.models import ChatRequest, ChatResponse

#user-frendly message When groq rate limit (daily token quota) in exceeded.
RATE_LIMIT_MESSAGE = (
    "you've reached your daily API limit for this assistant."
    "your credits will reset in a few hours, or you can Upgrade Your Plan for more"
    "Please try agen later."
)

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg

from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatServices
from app.config import VECTOR_STORE_DIR

from langchain_community.vectorstores import FAISS

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("S.Y.L.P.H")

# -----------------------------------------------------------------------------
# GLOBAL SERVICE REFERENCES
# -----------------------------------------------------------------------------

vector_store_service: VectorStoreService = None
groq_service : GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatServices = None

def print_title():
    """Print the S.Y.L.P.H ASCII art title."""
    title = """

         ______   ___     ____  _   _ 
        / ___\ \ / / |   |  _ \| | | |
        \___ \\ V /| |   | |_) | |_| |
         ___) || | | |___|  __/|  _  |
        |____/ |_| |_____|_|   |_| |_|
                        
                                                                                                                    
    """

    print(title)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """"""
    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()
    logger.info("="*60)
    logger.info("S.Y.L.P.H - Starting Up...........")
    logger.info("="*60)

    try:
        logger.info("Initializing vector store service.........")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vactor_store()
        logger.info("verctor store Initialized Successfully")

        logger.info("Initializing Groq service (general Querires).......")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service Initialized successfully")

        logger.info("Initializing realtime groq service (with tavil search)......")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq services Initialized Successfully")

        logger.info("Initializing chat Service...........")
        chat_service = ChatServices(groq_service, realtime_service)
        logger.info("Chat service Initialized Successfullly")

        logger.info("="*60)
        logger.info("Service status:")
        logger.info("   - Vector Store: Ready")
        logger.info("   - Groq AI (Gerneal): Ready")
        logger.info("   - Groq AI (Realtime): Ready")
        logger.info("   - Chat Service: Ready")
        logger.info("="*60)
        logger.info("S.Y.L.P.H is online and ready")
        logger.info("API http://localhost:8000")
        logger.info("="*60)

        yield

        logger.info("\nShatting down S.Y.L.P.H..............")
        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)
        logger.info("All session saved. goodbey!")
    
    except Exception as e:
        logger.error(f"Fital error during startup: {e}",exc_info=True)
        raise

app = FastAPI(
    title="S.Y.L.P.H API",
    description="Just A Rather very Intelligent System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ CORRECT
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "S.Y.L.P.H API",
        "endpoints": {
            "/chat": "General chat (pure LLM, no web search)",
            "/chat/realtime": "Realtime chat (with tavil search)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System Health check"
        }
    }

@app.get("/health")
async def health():
    return {
        "status":"healthy",
        "vector_store": vector_store_service is not None,
        "groq_service": groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service": chat_service is not None
    }
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service Not Initialized")

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_message(session_id, request.message)
        chat_service.save_chat_session(session_id)

        return ChatResponse(response=response_text, session_id=session_id)

    except ValueError as e:
        logger.warning(f"Invalid session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limit hit: {e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error(f"Error Processing Chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

    
@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="chat service not initialized")
    
    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime Service Not initialized")
    
    try:
        sesstion_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_realtime_message(sesstion_id, request.message)
        chat_service.save_chat_session(sesstion_id)
        return ChatResponse(response=response_text, session_id=sesstion_id)
    
    except ValueError as e:
        logger.warning(f"Invaild session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limit hit:{e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error(f"Error Process realtime chat:{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"error Processing chat: {str(e)}")
    
@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service no initialized")
    
    try:
        message=chat_service.get_chat_history(session_id)
        return {
            "session_id":session_id,
            "messages": [{"role": msg.role, "content":msg.content} for msg in message]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error retrieving history: {str(e)}")
    

def run():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run()