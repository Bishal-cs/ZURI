"""
CHAT SERVICE MOUDULE
====================

This services name all chat session and conversation logic.It is userd by the
/chat and /chat/realtime endpoins. Designed for single-user one server
hat use ChatService and one in memory seisson store; the user can save many
session (each identified by session_id).

RESPONSIBILIETS:
    -get_or_create_session(session_id): Return ecsisting session or create new one.
    if the user sends a session_id that was used before (e.g, before a restart),
    we try to load if from disk so the conversation continues.
    -add_message / get_chat_history: Keep messages in memory per session.
    -format_history_for_llm: Turn the messages list into (user, assistant) pairs
    and trim to MAX_CHATS_HISTROTY_TURNS so we don't overflow the prompt.
    -process_message / process_realtime_message: Add user message, call Groq (or
    RealtimeGroq), add assistant reply, return reply.
    -save_chat_session: Write session to database/chats_data/".json so it persists
    and can be loaded on next startup (and used by the vector store for retrieval).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from app.config import CHATS_DATA_DIR , MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
apps = CHATS_DATA_DIR

logger = logging.getLogger("S.Y.L.P.H.")

# =================================================================================
# CHAT SERVICES CLASS
# =================================================================================

class ChatServices:
    """
    Message chat sessions: in memory message lists, load,save to disk, and
    calling Groq (or Realtime) To get replies. ALL state for active sessions
    is in safe.sessions; saving to disk is done after message so
    conversation survive restarts.
    """

    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """Store references to the Groq and Realtime service; keep Session in Memory."""
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        # Map: Session_id -> List of Chatmessage (user and assistant message in order).
        self.sessions: Dict[str, List[ChatMessage]] = {}

# ---------------------------------------------------------------------------
# SESSION LOAD / VALIDATE /GET-OR-CREATE
# --------------------------------------------------------------------------- 
    def load_session_from_disk(self,session_id: str) -> bool:
        """
        Load a Session From database/chats_data/ if a file for this session_id exists.

        File name is chat_{safe_session_id}.json where safe_session_id has dashes/spaces removed.
        on success we put the messages into self.session{session_id} so later request use them.
        Return ture if loaded, False if file missing unreadable.
        """      

        # Sanitize Id For use Filename (no_dashes or spaces)
        safe_session_id = session_id.replace("-", "").replace(" ","_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename

        if not filename.expandtabs():
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
            # Convert stored dicts back ChatMessage Ogjects.
            messages = [
                ChatMessage(role=msg.get("role"), content=msg.get("content"))
                for msg in chat_dict.get("messages", [])
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning("Failed to load session %s from Disk: %s", session_id, e)
            return False
    def Validate_session_id(self, session_id: str) -> bool:
        """
        ReTurn True if session_id is safe to use (non-empty, no path traversal, length  <= 255).
        used To reject malilcious or invalid IDs before we use im file paths.
        """
        if not session_id or not session_id.strip():
            return False
        # Black Path Traversal and path separators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
    
    #def get_or_create_session(self, session_id: Optional[str] = None) -> str:
    def get_or_create_session(self, request, session_id=None):
        """
        ReTurn a Session ID and ensure that session exists in memory.

        -if session_id is None: Create a new session with a new UNID and return it.
        -if session_id is Provided: validate it; if's self.session return if;
            else try to load form disk; if not found, create a new session with the ID.
        Raises ValueError if session_id is invalid (empty, path traversal, or too long).
        """
        
        if not session_id:
            new_sessions_id=str(uuid.uuid4())
            self.sessions[new_sessions_id] = []
            return new_sessions_id
        
        

        if not self.Validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id format: {session_id}. session Id must be non-empty, "
                "not contation path traveral chatacters, and to under 255 charachares."
            )
        if session_id in self.sessions:
            return session_id
        
        if self.load_session_from_disk(session_id):
            return session_id
        
        # New session with this ID (e.g Client sent an ID that was naver saved).
        self.sessions[session_id] = []
        return session_id
    
    # -------------------------------------------------------------------------
    # Messages and history formating
    # -------------------------------------------------------------------------

    def process_chat(self, request, session_id):
        # implement your chat handling logic here
        # e.g., parse request, route to model, return response
        return "Chat response text"

    def add_message(self, session_id: str, role: str, content: str):
        """Append one message (user or assistant) to the session's message list. Creates session if missing."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))
    
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """ReTurn the list at message for this session (Chronological). Empty list if session unknown."""
        return self.sessions.get(session_id)
    
    def format_history_list(self, session_id: str, exclude_last:bool = False) -> List[tuple]:
        """
        build a list of (user_text, assisitant_text) parirs for the LLM prompt.

        we only include complete parits aand cap at MAX_CHAT_HISTORY_TURNS (e.g. 20)
        so the prompt does not grow unbeunded. if exclude_last is True we drop the
        last message (the current user message that we are about is reply to).
        """
        messages = self.get_chat_history(session_id)
        history = []
        # if exclude_last, we skip the last message (the current user message we are about reply to).
        messages_to_process = messages[:-1] if exclude_last and messages else messages
        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i + 1]
            if user_msg.role == "user" and ai_msg.role == "assistant":
                history.append((user_msg.content, ai_msg.content))
                i += 2
            else:
                i += 1
        # Keep only the most recent turns so the does not exceed token limilts.
        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]
        return history
    

    # ---------------------------------------------------------------------------
    # Process Message (general and realtime)
    # ---------------------------------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one gerneal-chat message: add user message, call Groq (no web search), add reply, return it.
        """
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_list(session_id, exclude_last=True)
        # responses = self.groq_service.get_response(
        #     user_message=user_message,
        #     chat_history=chat_history
        # )
        responses = self.groq_service.get_response(
            user_message,
            chat_history
        )


        return responses
    
    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one realtime message: add user message, call realtime service (Tavily + Groq), add reply, return it.
        Uses The same session as process_message so histroy is shared. Raises ValueError if realtime_service is None.
        """

        if not self.realtime_service:
            raise ValueError("Realtime Sevice is not initiallized. connot process realtime Queries")
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_list(session_id, exclude_last=True)
        response = self.realtime_service.get_response(user_message,chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        return response
    
    # -------------------------------------------------------------------------------
    # Presist sesstion to dise
    # -------------------------------------------------------------------------------

    def save_chat_session(self, session_id: str):
        """
        Write this session's messages to database/chats_data/chat_{safe_id}.json.

        called after each message so the conversation to persisted. The vector store
        is rebuild on startup from these files, so new chats are included after restart.
        if the sesion is missing or empty we do nothing. On write error as only log.
        """
        if session_id not in self.sessions or not self.sessions[session_id]:
            return
        
        messages = self.sessions[session_id]
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        chat_dict = {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content":msg.content} for msg in messages]
        }

        try:
            with open(filepath, "w",encoding="utf-8") as f:
                json.dump(chat_dict,f,indent=2,ensure_ascii=False)
        except Exception as e:
            logger.error("Falied to save chat session %s to disk: %s", session_id,e)
