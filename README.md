# 🎮 Clutch

🎙️ **Clutch** is a real-time voice assistant and tactical AI coach for Counter-Strike 2.

It listens to your voice, analyzes your game state (via GSI + screenshots), and responds with round-winning advice using AI models — all spoken back in real time.

---

## 🎥 Demo

[<video src="clutch_demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/24c870cd-2311-4cff-87d2-29ada9d58f13)

---

## ✨ Features

- 🔊 Wakeword detection — just say **"Hey Jarvis"**
- 🎙️ Real-time speech transcription (STT)
- 🧠 Tactical advice powered by streaming LLMs
- 📸 GSI + screenshot parsing for contextual awareness
- 🗣️ Coqui XTTS playback for ultra-fast voice responses

---

## 📁 Project Structure

```
clutch/
├── clutch_loop.py                      # Main runtime loop (voice → context → AI → voice)
├── requirements.txt                    # All pip packages required (generated via pip freeze)
├── models/                             # TTS & STT models (must download via Google Drive)
│   ├── Coqui/                          # XTTS model files
│   └── tiny.en/                        # OpenWakeWord + STT model
├── gamestate_integration_clutch.cfg    # GSI config (auto-copied into CS2 directory)
├── venv/                               # Pre-installed Python environment
│   └── ...                             # Contains all required packages
├── build_clutch.py                     # Optional: used to bundle into a `.bat` app
├── embedded_python/                    # Optional: portable Python
├── .env.example                        # Example environment file for storing API keys
```

---

## ⚙️ Setup Instructions (Windows Only)

### 1. Clone the repo

```bash
git clone https://github.com/hoangtu0701/clutch.git
cd clutch
```

### 2. Set up Python environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This installs everything exactly as used in development (from `pip freeze`).

### 4. Set up your API keys

```bash
# Copy and rename the example env file to env and edit it with your own keys
copy .env.example .env

# Then open `.env` and paste your API keys like:
OPENAI_API_KEY=sk...
OPENROUTER_API_KEY=sk...
```

---

## 📦 Download Required Models & Optional Python Runtime

The following folders are **not included** in the GitHub repo due to size limits.

📥 Download from Google Drive:  
👉 [Clutch folders](https://drive.google.com/drive/folders/1wAPdx7JF7OL3bMVblcqT-djFfrwNv_vB?usp=sharing)

You’ll get:

```
models/
├── Coqui/       # Coqui XTTS model files (e.g., model.pth, config.json)
├── tiny.en/     # Wakeword + transcription models
```

Optionally:

```
embedded_python/ # Used in build_clutch.py for portable app building
```

Place these folders inside your project directory.

---

## 🚀 Run Clutch

Once everything is installed and models are in place:

```bash
python clutch_loop.py
```

Say **“Hey Jarvis”**, then start talking. Clutch will:

- Capture your speech + game context (CS2 GSI + screenshot)
- Analyze everything with LLMs
- Respond instantly with tactical voice coaching

---

