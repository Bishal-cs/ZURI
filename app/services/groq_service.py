"""
GROQ SERVICE MODULE
===================

This Module Handel general chat: no web search, only the Groq LLM plus contest
From the vector store (learning datas + Pasts chats).Used By ChatService for
POST /chat.

MULIPLE API KEYS (round-robin and fallback):
    - You can Set multiple Groq API Key in .evn: Groq_api_key, Groq_api_key_2,
        Groq_api_key_3, ....(no limlit).
    - Each request uses one key in rotation: 1st request -> 1st key, 2nd request->
        2nd key. 3rd request -> 3rd key, then back to 1st key, and so on. Every key
        if used one-by-one so usage is spread across keys.
    - The Round-robin counter is shared across all instances (Groqservice and
        RealtimeGroqService), so Both /chat and /chat/realtime endpoint use the 
        same rotaion seaqunce.
    - If the Chouses key fails (rate limit 429 or any error), we try the next key,
        then the next, until one sucesseds or all have been tried.
    - All API key Usage in logged with maskef keys (first 8 and last 4 chars visible)
        for security and debugging purposes.
Follow:
    1. get_response(question, chat_histroy) is called
    2. we ask the vector store for the top-4 chunks most similar to the question (retrievel).
    3. we build a system message: SYLPH_SYSTEM_PROMPT + current time + retrieved context.
    4. we send to groq using the next key in rotation (or fallback to next key on failure).
    5. we return the assistant's reply.

Context is only what as retrieve (no full dump of learing data), so token usage stays bounded.

"""



from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging


from app.config import GROQ_API_KEYS, GROQ_MODEL, SYLPH_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information



logger = logging.getLogger("S.Y.L.P.H.")

# ==========================================================================
# HELPER: ESCAPE CURLY BRACES FOR LANGCHAIN
# ==========================================================================
# langChain prompt templates use {variable_name}. if learninig data or chat
# content contatins { or }, the template engine can break. Double then
# makes them literal in the final string.

def escape_curly_braces(text: str) -> str:
    """
    double every { and } so LangChain does not them as Template Variables.
    Learning data or chat content might contain { or }; without escaping, invoke() can fail.
    """

    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")

def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Return True if the exception indicates a Groq limit (e.g: 429, tokens per day).
    Used for logging; actual fallback tries the next key on any failure when multiple keys exist.
    """

    msg= str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg

def _mask_api_key(key: str) -> str:
    """
    Mask an Api key for safe logging. shows first B and Last 4 Characters, masks the middle.
    Ex:- gsk_123456acbseseedf -> gsk_12234...sdef
    """

    if not key or len(key) <= 11:
        return "***masked***"
    return f"{key[:8]}...{key[-4]}"

# ===============================================================================
# GROQ SERVICE CLASS
# ===============================================================================

class GroqService:
    """
    Genernal chat: retrieves context from the vector store and calls the Groq LLM.
    Support Muililple API Keys: Each Request uses the next in rotation (one-by-one),
    and if that key fails, the server tries next key until one succeeds or all fail.

    Round-bobin behavior:
     - Request 1 uses key 0 (first key)
     - Request 2 uses key 1 (second key)
     - Request 3 uses key 2 (thrid key)
     After all keys are used, cycles back to key 0
     - if a key fails (rate limit, error), tries the next key in sequence
     - All request share the same round-robin counter (class-level)
    """

    # Class-level counter shared across all instances {GroqService and RealtimeGroqService}
    # This Ensure round-robin works across both /chat and /chat/realtime endpoints
    _shared_key_index = 0
    _lock = None # will be set to threading lock if threading is needed (currently, single-threaded)

    def __init__(self, vector_store_service: VectorStoreService):
        """
        Create one Groq LLM clinet per API key store the vactor store for reatriveal.
        self.llms[i] corresponds to GROQ_API_KEYS[i]; request N uses key at index (N % len(keys)).
        """
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configur. set GROQ_API_KEY (and optinally, GROQ_API_KEY_2, GROQ_API_KEY3,....) in .evn"
            )
        # One ChatGroq instance per key; each request will use these in rotation.
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.8,
            )
            for key in GROQ_API_KEYS
        ]
        self.vector_store_service = vector_store_service
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s)")

    def _invoke_llm(
            self,
            prompt: ChatPromptTemplate,
            messages: list,
            question: str,
    ) -> str:
        """
        Call the LLM using the next in rotation; on failure, try the next key until one success.

        - Round-rabin: the Reques uses key index (_shared-key_index %n), then we increment
            _shared_key_index so the next request uses next key. ALL Instances Share the Same Counter.
        - Fallback: if the sucessfully or we have tried all keys.
        Return respones.contant. Reises if all keys fail.
        """

        n = len(self.llms)
        # Which key try First for this request (round-rabin: request 1 -> key 0, request 2 -> key 1, ...).
        # Use Class-level counter so all instances (GroqService and RealtimeGroqService) Share the same rotation.
        start_i = GroqService._shared_key_index % n
        current_key_index = GroqService._shared_key_index
        GroqService._shared_key_index += 1

        # Log Which key we're using (masked for security)
        masked_key = _mask_api_key(GROQ_API_KEYS[start_i])
        logger.info(f"Using API key #{start_i + 1}/ {n} (round-robin index: {current_key_index}): {masked_key}")

        last_exc = None
        keys_tried = []
        #try each key in order starting form start_i (wrap around with %n)
        for j in range(n):
            i = (start_i + j) % n
            keys_tried.append(i)
            try:
                #Build Chain With Key's LLM and Invoke once.
                chain = prompt | self.llms[i]
                response = chain.invoke({"history": messages, "question": question})
                #Log Sueccess if we had to fallback to a different key
                if j > 0:
                    makeked_seccess_key = _mask_api_key(GroqService[i])
                    logger.info(f"Fallback Sucessfull API Key #{i + 1}/{n} sucesseded: {makeked_seccess_key}")
                return response.content
            except Exception as e:
                last_exc = e
                makeked_Falied_key = _mask_api_key(GROQ_API_KEYS[i])

                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} race limlited: {makeked_Falied_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} falied: {makeked_Falied_key} - {str(e)[:100]}")
                # if we have more than one key, try the next one; otherwise rasise inmmediatelay.
                if n > 1:
                    continue
                raise Exception(f"Error Getting resposes For Groq: {str(e)}") from e
        # All keys ware tried and all failed; raise the last exection.
        masked_all_keys = ", ".join([_mask_api_key(GROQ_API_KEYS[i]) for i in keys_tried])
        logger.error(f"All api keys failed. tried keys: {masked_all_keys}")
        raise Exception(f"Error getting response From Groq: {str(last_exc)}") from last_exc
    
    def get_response(
            self,
            quescation: str,
            chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        Return the Assistant's replay for this question (genral chat, on web search).
        Retrieves context from the vecoter store, the prompt, then calls _invalid_llm
        Which uses the next api key in rotation and falls back to other keys on faliure.
        """

        try:
            # Get relavent chunks from learning data and past chats {bounded token usage}.
            # if retrilevel falis (e.g vector store not reday), use empty context so the llm still answers.
            context = ""
            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_does = retriever.invoke(quescation)
                context = "\n".join([doc.page_content for doc in context_does]) if context_does else ""
            except Exception as retrieval_err:
                logger.warning("Vectory store resrtieval failed, using empty context: %s", retrieval_err)
            
            # Build system Message: Personality + currnet time + retrieved context.
            time_info = get_time_information()
            system_message = SYLPH_SYSTEM_PROMPT + f"\n\nCurrent time and date: {time_info}"
            if context:
                system_message += f"\n\nRelvenet Context from your learing data and past conversations:\n{escape_curly_braces(context)}"
            
            # Prompt Templte: system Message, Chat History Placeholder, current question.
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human","{question}"),
            ])
            # Converst (user assistent) apirs to LangChain Message Objects.
            message = []
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    message.append(HumanMessage(content=human_msg))
                    message.append(AIMessage(content=ai_msg))

            # Use Next key in rotation; on failure, try remaining keys (same as realtime).
            return self._invoke_llm(prompt, message, quescation)
        except Exception as e:
            raise Exception(f"Errpr Getting response from Groq: {str(e)}") from e
        
        