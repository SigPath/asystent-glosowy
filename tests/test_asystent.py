"""Testy asystenta — pamięć konwersacji, pipeline, komponenty Echo. Bez sieci i audio."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asystent.brain import SYSTEM_PROMPT  # noqa: E402
from asystent.components import EchoLLM, EchoSTT, SilentTTS  # noqa: E402
from asystent.memory import respond  # noqa: E402
from asystent.pipeline import Pipeline  # noqa: E402


class TestMemory(unittest.TestCase):
    def test_pamiec_rosnie_i_zawiera_system(self):
        memory = []
        reply, memory = respond("hej", memory, EchoLLM())
        self.assertEqual(reply, "(echo) hej")
        self.assertEqual(len(memory), 2)  # user + assistant
        _, memory = respond("co dalej", memory, EchoLLM())
        self.assertEqual(len(memory), 4)

    def test_pamiec_ograniczona_do_20(self):
        memory = []
        for i in range(30):
            _, memory = respond(f"msg {i}", memory, EchoLLM())
        self.assertLessEqual(len(memory), 20)


class TestPipeline(unittest.TestCase):
    def test_text_turn_z_tts(self):
        tts = SilentTTS()
        pipe = Pipeline(stt=EchoSTT("co umiesz"), llm=EchoLLM(), tts=tts)
        reply = pipe.text_turn("co umiesz")
        self.assertEqual(reply, "(echo) co umiesz")
        self.assertEqual(tts.spoken, ["(echo) co umiesz"])

    def test_listen_turn_przechodzi_przez_stt(self):
        pipe = Pipeline(stt=EchoSTT("pytanie z mikrofonu"), llm=EchoLLM(), tts=SilentTTS())
        self.assertEqual(pipe.listen_turn("fake.wav"), "(echo) pytanie z mikrofonu")

    def test_reset_czysci_pamiec(self):
        pipe = Pipeline(stt=EchoSTT(), llm=EchoLLM(), tts=SilentTTS())
        pipe.text_turn("jedna")
        pipe.reset()
        self.assertEqual(pipe.memory, [])

    def test_system_prompt_wchodzi_do_llm(self):
        seen = {}

        class SpyLLM:
            def chat(self, messages):
                seen.update(messages=messages)
                return "ok"

        pipe = Pipeline(stt=EchoSTT(), llm=SpyLLM(), tts=SilentTTS())
        pipe.text_turn("test")
        self.assertEqual(seen["messages"][0]["role"], "system")
        self.assertEqual(seen["messages"][0]["content"], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
