"""Komponenty: STT, LLM, TTS. Wersje rzeczywiste (Ollama/Whisper/Piper) + echo/silent do testów."""
import json
import subprocess
import urllib.request
from typing import List, Protocol, Tuple


class STT(Protocol):
    def transcribe(self, wav_path: str) -> str: ...


class LLM(Protocol):
    def chat(self, messages: List[dict]) -> str: ...


class TTS(Protocol):
    def say(self, text: str) -> None: ...


class WhisperSTT:
    """faster-whisper (lokalnie). Zależność opcjonalna: pip install faster-whisper."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def transcribe(self, wav_path: str) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, compute_type="int8")
        segments, _ = self._model.transcribe(wav_path, language="pl")
        return " ".join(s.text.strip() for s in segments).strip()


class EchoSTT:
    """Zwraca tekst pliku „promptu” — do testów."""

    def __init__(self, fixed: str = "jakie masz możliwości"):
        self.fixed = fixed

    def transcribe(self, wav_path: str) -> str:
        return self.fixed


class OllamaLLM:
    """LLM przez lokalną Ollamę (/api/chat). Bez zależności."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages: List[dict]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("message", {}).get("content", "")


class EchoLLM:
    """Deterministyczny LLM do testów: echa ostatniej wiadomości użytkownika."""

    def chat(self, messages: List[dict]) -> str:
        last = [m for m in messages if m["role"] == "user"]
        return f"(echo) {last[-1]['content'] if last else '…'}"


class PiperTTS:
    """Piper TTS przez subprocess: piper --model <onnx> > plik.wav, potem odtwarzacz systemowy."""

    def __init__(self, model_path: str, player: Tuple[str, ...] = ("afplay",)):
        self.model_path = model_path
        self.player = player  # macOS: afplay, Linux: aplay / paplay

    def say(self, text: str) -> None:
        piper = subprocess.Popen(
            ["piper", "--model", self.model_path, "--output_file", "/tmp/asystent_tts.wav"],
            stdin=subprocess.PIPE,
        )
        piper.communicate(text.encode("utf-8"))
        subprocess.run([*self.player, "/tmp/asystent_tts.wav"], check=False)


class SilentTTS:
    """Nie mówi — zapisuje, co miałoby zostać powiedziane (testy/demo)."""

    def __init__(self):
        self.spoken: List[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
