"""Pipeline: jedno „uszy → mózg → usta” w jednym obiekcie, z historią konwersacji."""
from typing import List

from .brain import SYSTEM_PROMPT
from .memory import respond


class Pipeline:
    def __init__(self, stt, llm, tts):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.memory: List[dict] = []

    def listen_turn(self, wav_path: str) -> str:
        """Nagranie → tekst → odpowiedź → (opcjonalnie) mowa. Zwraca tekst odpowiedzi."""
        user_text = self.stt.transcribe(wav_path)
        return self.text_turn(user_text)

    def text_turn(self, user_text: str) -> str:
        reply, self.memory = respond(user_text, self.memory, self.llm)
        self.tts.say(reply)
        return reply

    def reset(self) -> None:
        self.memory = []
