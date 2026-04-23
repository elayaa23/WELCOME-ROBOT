from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from ollama import chat

stt_model = get_stt_model()
tts_model = get_tts_model()

_is_processing = False

def echo(audio):
    global _is_processing
    if _is_processing:
        return
    _is_processing = True
    try:
        transcript = stt_model.stt(audio)
        if not transcript or not transcript.strip():
            return
        response = chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": transcript}],
            options={"num_predict": 60},
            stream=True,
        )
        buffer = ""
        for chunk in response:
            buffer += chunk["message"]["content"]
        for audio_chunk in tts_model.stream_tts_sync(buffer.strip()):
            yield audio_chunk
    finally:
        _is_processing = False

stream = Stream(ReplyOnPause(echo, can_interrupt=True), modality="audio", mode="send-receive")
stream.ui.launch()
