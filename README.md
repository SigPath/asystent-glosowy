# Głosowy asystent offline

**„Siri bez chmury”**: mowa → tekst → lokalny LLM → mowa. Wszystko na Twoim komputerze:
whisper (STT) + Ollama (LLM) + Piper (TTS). Zero wysyłanych nagrań, zero kosztów API.

## Architektura

```
mikrofon ──▶ STT (whisper) ──▶ mózg (Ollama + pamięć 20 wiadomości) ──▶ TTS (Piper) ──▶ głośnik
```

```
asystent/
├── components.py  # STT/LLM/TTS: wersje rzeczywiste + Echo/Silent do testów
├── brain.py       # systemowy prompt PL (zwięźle, bez zmyślania, format pod mowę)
├── memory.py      # respond(): pamięć konwersacji z limitem 20 wiadomości
├── pipeline.py    # Pipeline: listen_turn / text_turn / reset
├── audio.py       # nagrywanie mikrofonu (sounddevice, opcjonalne)
└── cli.py         # demo | text | speak
```

Każdy komponent ma kontrakt (Protocol) — zamiana whispera na inny STT to jedna klasa.

## Szybki start

```bash
# demo bez żadnych zależności (komponenty Echo — pokazuje przepływ i pamięć):
python3 -m asystent demo

# jedna tura z tekstem (wymaga działającej Ollamy):
python3 -m asystent text "masz jakieś pomysły na kolację?"

# pełna pętla głosowa (wymaga: pip install faster-whisper sounddevice numpy scipy + piper):
python3 -m asystent speak --piper-model pl_PL-darkman-medium.onnx
```

## Testy

```bash
python3 tests/test_asystent.py   # 6 testów: pamięć+limit, pipeline, TTS hook, system prompt
```

Testy wykryły realny bug: pamięć rosła bez limitu mimo cięcia kontekstu dla LLM — poprawione.

## Uczciwe ograniczenia

- pętla `speak` nagrywa stałe odcinki (6 s); wyznaczanie początku/końca mowy (VAD) w planach,
- latencja zależy od sprzętu: whisper „small” + llama3.2 3B ≈ kilka sekund na Apple Silicon,
- Piper wymaga jednorazowego pobrania modelu głosu PL (np. pl_PL-darkman-medium).

## Licencja

MIT
