"""Nagrywanie audio przez sounddevice (opcjonalna zależność) do pliku WAV."""
from typing import Optional


def record_wav(path: str, seconds: float = 6.0, samplerate: int = 16000) -> Optional[str]:
    """Nagrywa z domyślnego mikrofonu. Wymaga: pip install sounddevice numpy scipy.

    Zwraca ścieżkę pliku albo None, gdy zależności/urządzenie nie są dostępne.
    """
    try:
        import sounddevice as sd
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        return None
    try:
        audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        wavfile.write(path, samplerate, np.asarray(audio))
        return path
    except Exception:
        return None
