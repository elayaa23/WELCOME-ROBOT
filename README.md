
# 🚀 Local Voice AI Agent

## 🧠 Overview

This project implements a **fully local voice AI pipeline** designed for robotic and industrial applications.

### 🔁 Pipeline

- 🎤 **Speech-to-Text (STT)** → Faster-Whisper  
- 🧠 **Language Model (LLM)** → Ollama (Gemma 1B)  
- 🔊 **Text-to-Speech (TTS)** → Kokoro  
- 📚 **Optional RAG** → ChromaDB + local documents  

➡️ Runs **100% locally (offline)** → secure & suitable for industrial use

---

## 📁 Project Structure

```

local-voice-ai-agent/
│
├── local_voice_chat.py
├── local_voice_chat_advanced.py
├── local_voice_chat_rag.py
├── rag_setup.py
├── documents/
├── README.md
├── pyproject.toml
├── .gitignore

````

---

## ⚙️ Setup (WSL / Linux)

### 1. Open WSL terminal

* wsl -d Ubuntu

* cd local-voice-ai-agent
````

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Ollama

In WSL:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then run:

```bash
ollama run gemma3:1b
```
ollama run gemma3:4b (the advanced version)
---

### 5. Start Kokoro TTS server

Make sure Kokoro is running:

Example:

```bash
python kokoro_server.py
```

(Or your existing FastAPI server)

---

## ▶️ Run

### 🔹 Basic

```bash
python local_voice_chat.py
```

---

### 🔹 Advanced

```bash
python local_voice_chat_advanced.py
```

---

### 🔹 RAG

```bash
python rag_setup.py
'python local_voice_chat_rag.py'
```

---

## ⚠️ Notes

* CPU execution → **higher latency**
* Fully offline → **more secure but slower**
* Requires:

  * Ollama running
  * Kokoro server running

---

## 🔧 What has been implemented

* ✔️ Full **local STT → LLM → TTS pipeline**
* ✔️ Multiple system versions (basic / advanced / RAG)
* ✔️ RAG with **ChromaDB**
* ✔️ Functional demo interface
* ✔️ Tested end-to-end interaction

---

## 🔮 Future Improvements

* 🔗 ROS2 / Isaac Sim integration
* ⚡ Latency optimization
* 🤖 Real robot interaction
* 🧠 Smarter dialogue handling / Multimodality


---

## 🎯 Goal

Foundation for a **voice-enabled robot assistant** capable of:

* Natural interaction
* Context-aware responses
* Offline deployment

---

## 👩‍💻 Author

Aya El Alaoui Najib

```

---

# ⚠️ SUPER IMPORTANT (WSL SPECIFIC)


👉 If audio doesn’t work:
- WSL microphone access can be tricky  
- you may need to run on **Windows Python instead**

