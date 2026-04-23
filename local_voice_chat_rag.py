import sys
import queue
import threading
import argparse
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from ollama import chat

stt_model = get_stt_model()
tts_model = get_tts_model()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

DB_DIR = "chroma_db"
TOP_K = 3

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    total = vector_db._collection.count()
    logger.info(f"📚 Loaded RAG database: {total} chunks")
except Exception as e:
    logger.warning(f"⚠️  Could not load RAG database: {e}")
    vector_db = None

def retrieve_context(query):
    if vector_db is None or not query.strip():
        return ""
    try:
        results = vector_db.similarity_search(query, k=TOP_K)
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "")
            ref = f"{source} p.{page}" if page != "" else source
            parts.append(f"[{i}] ({ref})\n{doc.page_content.strip()[:300]}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"RAG retrieval error: {e}")
        return ""

SYSTEM_PROMPT = """You are a helpful voice assistant with access to a knowledge base.
Use the context provided to give accurate answers.
If the context is not relevant, answer from general knowledge.
Reply in plain text only, no emojis or markdown.
Keep answers to 2 to 3 sentences."""

_DONE = object()

def llm_worker(user_message, audio_queue):
    try:
        response_stream = chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"num_predict": 60},
            stream=True,
        )
        buffer = ""
        for chunk in response_stream:
            buffer += chunk["message"]["content"]
        response_text = buffer.strip()
        logger.debug(f"🤖 Response: {response_text}")
        if response_text:
            for audio_chunk in tts_model.stream_tts_sync(response_text):
                audio_queue.put(audio_chunk)
    except Exception as e:
        logger.error(f"LLM worker error: {e}")
    finally:
        audio_queue.put(_DONE)

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
        context = retrieve_context(transcript)
        if context:
            logger.debug(f"📖 Retrieved {TOP_K} chunks")
            user_message = f"Context:\n{context}\n\nQuestion: {transcript}"
        else:
            logger.debug("📭 No context — general knowledge.")
            user_message = transcript
        audio_queue = queue.Queue()
        t = threading.Thread(target=llm_worker, args=(user_message, audio_queue), daemon=True)
        t.start()
        while True:
            item = audio_queue.get(timeout=30)
            if item is _DONE:
                break
            yield item
    except queue.Empty:
        logger.error("⏱ LLM timed out.")
    except Exception as e:
        logger.error(f"echo error: {e}")
    finally:
        _is_processing = False

def create_stream():
    return Stream(ReplyOnPause(echo, can_interrupt=True), modality="audio", mode="send-receive")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat with RAG")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--phone", action="store_true")
    args = parser.parse_args()
    stream = create_stream()
    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info(f"Launching Gradio UI {'(public link)' if args.share else '(local only)'}...")
        stream.ui.launch(share=args.share)
