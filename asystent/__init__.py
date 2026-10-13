"""Głosowy asystent offline: whisper (STT) + Ollama (LLM) + Piper (TTS). Zero chmury."""
from .components import OllamaLLM, EchoLLM, PiperTTS, SilentTTS, WhisperSTT, EchoSTT
from .brain import SYSTEM_PROMPT
from .memory import respond
from .pipeline import Pipeline

__version__ = "1.0.0"
__all__ = [
    "OllamaLLM", "EchoLLM", "PiperTTS", "SilentTTS", "WhisperSTT", "EchoSTT",
    "SYSTEM_PROMPT", "respond", "Pipeline", "__version__",
]
