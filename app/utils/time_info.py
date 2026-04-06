"""
TIME INFORMATION UTILITY
=========================

    Returns a short, readable string with the current date and time. This is
    injected into the system prompt so the LLM can answer "what day is it?"
    and similar questions. Called by both GroqService and RealtimeGroqService.
"""
from datetime import datetime

def get_time_information() -> str:
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y, %I:%M %p")