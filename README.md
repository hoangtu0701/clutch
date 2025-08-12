# 🎮 Clutch

🎙️ **Clutch** is a real-time voice assistant and tactical AI coach for Counter-Strike 2.

It listens to your voice, analyzes your game state (via GSI + screenshots), and responds with round-winning advice using AI models — all spoken back in real time.

---

## 🎥 Demo

https://github.com/user-attachments/assets/c0977d21-d7a4-49e0-9168-cdaa2423d8c9

---

## ✨ Features

- 🔊 Wakeword detection — just say **"Hey Jarvis"**
- 🎙️ Real-time speech transcription (STT)
- 🧠 Tactical advice powered by streaming LLMs
- 📸 GSI + screenshot parsing for contextual awareness
- 🗣️ **TTS pipeline with smart fallback**  
  - Primary: **Google Chirp 3 HD (streaming)** if a Google key is present  
  - Fallbacks: **Piper** (low-VRAM) or **Coqui XTTS v2** (high-VRAM) — chosen automatically
- ⏱ **In-round timer badge** with auto-detection for freezetime, live, bomb plant, and round end
- 🖼️ **Multi-region screen capture** (radar + alive counters) to boost tactical precision
- 🪟 **Desktop UI** (PyQt) with live badges for TTS engine and round timer
- 🗂️ **Secret keys folder** for simple, portable configuration
- 🛠️ **Auto GSI setup**: copies `gamestate_integration_clutch.cfg` to your CS2 `cfg` folder if missing

---

## 📁 Project Structure

```
clutch/
├── clutch_loop.py                      # Main runtime loop (voice → context → AI → voice)
├── requirements.txt                    # All pip packages required (generated via pip freeze)
├── models/                             # TTS & STT models (must download via Google Drive)
│   ├── Coqui/                          # XTTS model files
│   ├── Piper/                          # Piper runtime + voice (.onnx + .json)
│   └── tiny.en/                        # OpenWakeWord + STT model
├── gamestate_integration_clutch.cfg    # GSI config (auto-copied into CS2 directory)
├── secret_keys/                        # Store keys here (see setup)
│   ├── openai.key
│   ├── openrouter.key                  # optional (not used right now)
│   └── google_tts_key.json             # optional (enables Google TTS streaming)
├── venv/                               # To-be-installed packages & Python environment
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

### 4. Create your `secret_keys` folder

Create a folder at the project root named `secret_keys/` with up to **three** files:

```
secret_keys/
├── openai.key                # REQUIRED: paste your OpenAI API key (just the key string)
├── openrouter.key            # OPTIONAL: not needed right now
└── google_tts_key.json       # OPTIONAL: paste your service account JSON to enable Google TTS streaming
```

- If **`google_tts_key.json` is missing**, Clutch will **automatically** use local TTS: Piper (for low VRAM) or Coqui XTTS (for high VRAM).
- If **`openai.key` is missing**, the app cannot call the model and won’t work.

> ⏳ **Heads-up:** first startup can take a bit (model + audio warmup). Leave the window alone for ~30s on first run.

---

## 📦 Download Required Models

The following folder is **not included** in the GitHub repo due to size limits.

📥 Download from Google Drive:  
👉 [Clutch folders](https://drive.google.com/drive/folders/1wAPdx7JF7OL3bMVblcqT-djFfrwNv_vB?usp=sharing)

You’ll get:

```
models/
├── Coqui/       # Coqui XTTS model files (e.g., model.pth, config.json)
├── Piper/       # Piper runtime + a voice, e.g.:
│   ├── piper/piper.exe
│   ├── en_US-norman-medium.onnx
│   └── en_US-norman-medium.onnx.json
├── tiny.en/     # Wakeword + transcription models
```

Place the whole models/ folder inside your Clutch project directory.

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

> 💡 While Clutch is speaking, you can say **“Hey Jarvis”** again to interrupt and ask something new.

---

## 🧩 Notes & Troubleshooting

- **GSI auto-setup:** on first run, the app tries to copy `gamestate_integration_clutch.cfg` into your CS2 `cfg` folder. If it fails, run the app as admin and/or copy it manually.
- **TTS selection:** with no Google key, Clutch auto-picks **Piper** if your GPU has low VRAM (≈ <12GB) or **Coqui XTTS** if there’s plenty.
- **Round timer badge:** shows `mm:ss` and auto-switches to a bomb timer when planted.
- **Screenshots:** only captured if a visible “Counter-Strike 2” window is found on your desktop.

---
