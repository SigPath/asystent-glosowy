"""CLI asystenta:  python3 -m asystent demo | text "..." | speak"""
import argparse
import os
import sys
from typing import List

from .brain import SYSTEM_PROMPT
from .components import EchoLLM, EchoSTT, OllamaLLM, SilentTTS
from .pipeline import Pipeline


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="asystent",
        description="Głosowy asystent offline: whisper (STT) + Ollama (LLM) + Piper (TTS), wszystko lokalnie.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    text_p = sub.add_parser("text", help="jedna tura z tekstu (bez mikrofonu)")
    text_p.add_argument("message")
    text_p.add_argument("--model", default="llama3.2")
    text_p.add_argument("--host", default="http://localhost:11434")

    speak_p = sub.add_parser("speak", help="pętla głosowa: nagrywaj → odpowiadaj głosem")
    speak_p.add_argument("--piper-model", required=True, help="ścieżka do modelu .onnx Pidera")
    speak_p.add_argument("--seconds", type=float, default=6.0)
    speak_p.add_argument("--model", default="llama3.2")

    sub.add_parser("demo", help="scenariusz demonstracyjny na komponentach Echo")

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        tts = SilentTTS()
        pipe = Pipeline(stt=EchoSTT("hej, co potrafisz"), llm=EchoLLM(), tts=tts)
        print(f"SYSTEM: {SYSTEM_PROMPT[:70]}…")
        for user in ["hej, co potrafisz", "a czym się różnisz od Siri", "reset i jeszcze raz hej"]:
            if user.startswith("reset"):
                pipe.reset()
                print(f"\nUŻYTKOWNIK: {user} (pamięć wyczyszczona)")
            else:
                print(f"\nUŻYTKOWNIK: {user}")
            reply = pipe.text_turn(user)
            print(f"ASYSTENT: {reply}")
            print(f"(głos: {tts.spoken[-1]!r})")
        print(f"\nPamięć po turach: {len(pipe.memory)} wiadomości")
        return 0

    if args.cmd == "text":
        pipe = Pipeline(stt=EchoSTT(""), llm=OllamaLLM(model=args.model, host=args.host), tts=SilentTTS())
        print(pipe.text_turn(args.message))
        return 0

    if args.cmd == "speak":
        try:
            from .components import OllamaLLM, PiperTTS, WhisperSTT
            from .audio import record_wav
        except ImportError:
            print("BŁĄD: pip install faster-whisper sounddevice numpy scipy", file=sys.stderr)
            return 2
        import tempfile
        pipe = Pipeline(
            stt=WhisperSTT(model_size="small"),
            llm=OllamaLLM(model=args.model),
            tts=PiperTTS(model_path=args.piper_model),
        )
        print("Pętla głosowa (Ctrl+C aby wyjść). Mów po polsku…")
        while True:
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "wejscie.wav")
                print("\n…słucham…")
                if not record_wav(wav, seconds=args.seconds):
                    print("BŁĄD: mikrofon niedostępny.", file=sys.stderr)
                    return 2
                print(f"ASYSTENT: {pipe.listen_turn(wav)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
