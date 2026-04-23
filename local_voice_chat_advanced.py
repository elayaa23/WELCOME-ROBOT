import sys
import argparse
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

stt_model = get_stt_model()
tts_model = get_tts_model()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

_is_processing = False

def echo(audio):
    global _is_processing
    if _is_processing:
        logger.warning("⚠️ Already processing — skipping frame.")
        return
    _is_processing = True
    try:
        transcript = stt_model.stt(audio)
        if not transcript or not transcript.strip():
            logger.debug("🔇 Empty transcript, skipping.")
            return
        logger.debug(f"🎤 Transcript: {transcript}")
        response_stream = chat(
            model="gemma3:1b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Reply in plain text only, no emojis or markdown. Keep answers to 2 sentences.",
                },
                {"role": "user", "content": transcript},
            ],
            options={"num_predict": 60},
            stream=True,
        )
        buffer = ""
        for chunk in response_stream:
            buffer += chunk["message"]["content"]
        response_text = buffer.strip()
        logger.debug(f"🤖 Response: {response_text}")
        for audio_chunk in tts_model.stream_tts_sync(response_text):
            yield audio_chunk
    finally:
        _is_processing = False

def create_stream():
    return Stream(ReplyOnPause(echo, can_interrupt=True), modality="audio", mode="send-receive")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument("--phone", action="store_true")
    args = parser.parse_args()
    stream = create_stream()
    if args.phone:
        stream.fastphone()
    else:
        stream.ui.launch()
