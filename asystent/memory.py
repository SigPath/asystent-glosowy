"""Mózg konwersacji: pamięć wiadomości + jedna tura rozmowy (czysta logika)."""
from typing import List, Tuple

from .brain import SYSTEM_PROMPT


def respond(user_text: str, memory: List[dict], llm) -> Tuple[str, List[dict]]:
    """Dodaje turę użytkownika, pyta LLM o odpowiedź, dopisuje turę asystenta.

    Zwraca (odpowiedź, nowa_pamięć). Pamięć ograniczana do ostatnich 20 wiadomości,
    by kontekst nie rósł bez końca.
    """
    memory = list(memory) + [{"role": "user", "content": user_text}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory[-20:]
    reply = llm.chat(messages)
    memory = (memory + [{"role": "assistant", "content": reply}])[-20:]
    return reply, memory
