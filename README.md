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
- 🧠 Streaming LLM responses (GPT-5 by default; OpenRouter optional)
- 📸 GSI + screenshot parsing for contextual awareness
- 🕒 **Smart in-round timer** (freezetime/live/over + bomb) shown in the UI
- 🔁 **Dynamic TTS**: auto-picks **Piper** on low VRAM, switches to **Coqui XTTS** on higher VRAM

> On startup we check GPU VRAM. Under ~12 GB, Clutch uses **Piper** for smooth, low-overhead speech. With more VRAM, it switches to **Coqui XTTS** for higher quality.

---

## 📁 Project Structure

```
clutch/
├── clutch_loop.py                      # Main runtime loop (voice → context → AI → voice)
├── requirements.txt                    # Pip packages
├── models/
│   ├── Piper/
│   │   ├── piper/                      # Contains piper.exe (Windows)
│   │   │   └── piper.exe
│   │   ├── en_US-norman-medium.onnx
│   │   └── en_US-norman-medium.onnx.json
│   ├── Coqui/                          # (Optional) XTTS v2 model files for high-VRAM
│   └── tiny.en/                        # Wakeword + STT model(s)
├── gamestate_integration_clutch.cfg    # GSI config (auto-copied into CS2 on first run)
├── build_clutch.py                     # Optional: bundle helper
├── embedded_python/                    # Optional: portable Python runtime
├── .env.example                        # Example env for API keys
```

---

## ⚙️ Setup (Windows)

### 1) Clone

```bash
git clone https://github.com/hoangtu0701/clutch.git
cd clutch
```

### 2) Python env

```bash
python -m venv venv
venv\Scriptsctivate
```

### 3) Install deps

```bash
pip install -r requirements.txt
```

### 4) API keys

```bash
copy .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=sk-...
# (Optional) OPENROUTER_API_KEY=sk-...
```

> We default to OpenAI GPT-5. OpenRouter is optional for experimentation.

### 5) Models & binaries

Create this layout under `models/`:

```
models/
  Piper/
    piper/piper.exe
    en_US-norman-medium.onnx
    en_US-norman-medium.onnx.json
  Coqui/                # (optional, only if you want XTTS)
  tiny.en/
```

- **Piper**: put `piper.exe` in `models/Piper/piper/` and a voice pair (`.onnx` + `.json`) in `models/Piper/`.
- **Coqui XTTS** (optional): place your XTTS v2 model files in `models/Coqui/`.
- **Wakeword/STT**: place the small STT/wakeword model(s) in `models/tiny.en/`.

> Paths above match what the app expects at runtime.

---

## 🚀 Run Clutch

```bash
python clutch_loop.py
```

On first run:

1) **GSI config is auto-copied** into the CS2 cfg folder:  
   `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg`  
   → **Restart CS2** after this first run so it starts POSTing game state.
2) The model is **warmed up** (tiny request) so the first real response is faster.
3) The UI opens. You’ll see a **TTS engine** badge and a **⏱ round timer** badge at the top.

Then say **“Hey Jarvis”** and talk. Clutch will:

- Capture your speech + game context (CS2 GSI + small HUD crops)
- Stream tactical advice from the model
- Start speaking back **as tokens arrive** (no full-sentence wait)

---

## 🧪 Tips & Troubleshooting

- **No CS2 data?** Make sure CS2 is running in a normal window (not minimized) and you’ve restarted it after the first run so GSI activates.
- **TTS stutter on GPU?** Piper is chosen automatically on low VRAM. If you force Coqui and get glitches, switch back to Piper.
- **Black window / no screenshots?** The screen capturer needs the CS2 window to be visible.
- **First reply slow?** That’s the warm-up; subsequent replies are faster.

---
