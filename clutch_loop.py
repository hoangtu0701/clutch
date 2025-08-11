# Add 'libs' directory to sys.path
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(base_dir, "libs")
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import necessary libraries
import json
import threading
import shutil
import torch
import filecmp
import time
import base64
import signal
import cv2
import bettercam
import pygetwindow as gw
import traceback
import requests
import re 
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, CoquiEngine, PiperEngine, PiperVoice
from flask import Flask, request
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# ---------------------
# Clients & Usage Setup
# ---------------------

openai_client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# OpenRouter model registries (3 models for different tasks)
openrouter_model_ids = {
    "GPT 5": "openai/gpt-5-chat",
    "Llama": "meta-llama/llama-3.2-3b-instruct",
    "DeepSeek R1 0528": "deepseek/deepseek-r1-0528:free",
    "LLaMA 3.1 405B Instruct": "meta-llama/llama-3.1-405b-instruct:free"
}

# Function to check if an OpenRouter model is free
def is_openrouter_model_free(model_id):
    try:
        res = requests.get("https://openrouter.ai/api/v1/models")
        res.raise_for_status()
        models = res.json().get("data", [])
        for model in models:
            if model["id"] == model_id:
                pricing = model.get("pricing", {})
                prompt_cost = float(pricing.get("prompt", "1"))
                completion_cost = float(pricing.get("completion", "1"))
                return prompt_cost == 0.0 and completion_cost == 0.0
    except Exception as e:
        print("Failed to fetch model pricing:", e)
    return False

# Function to call OpenAI model
def call_openai(messages):
    try:
        stream = openai_client.responses.create(
            model="gpt-5-chat-latest",
            input=messages,
            temperature=0.5,
            stream=True
        )
        return stream
    except Exception as e:
        traceback.print_exc()
        print("OpenAI failed")
        return None

# Function to call OpenRouter modeL
def call_openrouter(model, messages, max_tok, stream_bool, temp):
    try:
        result = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tok,
            stream=stream_bool,
            temperature=temp
        )
        return result

    except Exception as e:
        traceback.print_exc()
        print("OpenRouter failed")
        return None



# --------------------
# CS2 GSI Config Setup
# --------------------
def setup_gsi_cfg():
    GSI_FILENAME = "gamestate_integration_clutch.cfg"
    CS2_CFG_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"
    GSI_SRC_PATH = os.path.join(os.getcwd(), GSI_FILENAME)
    GSI_DEST_PATH = os.path.join(CS2_CFG_DIR, GSI_FILENAME)

    # Only copy if file is missing or differs from source
    try:
        if not os.path.isfile(GSI_SRC_PATH):
            print(f"GSI config file not found at: {GSI_SRC_PATH}. Please ensure it exists.")
            return
        if os.path.exists(GSI_DEST_PATH) and filecmp.cmp(GSI_SRC_PATH, GSI_DEST_PATH, shallow=False):
            print(f"GSI config already exists at: {GSI_DEST_PATH}. No need to copy.")
            return
        shutil.copy(GSI_SRC_PATH, GSI_DEST_PATH)
        print(f"GSI config successfully copied to: {GSI_DEST_PATH}. Please restart CS2 to apply changes.")
    except PermissionError:
        print("Permission denied while copying GSI config. Try running this script as administrator.")
    except Exception as e:
        print(f"Failed to copy GSI config: {e}")



# ------------------------------
# GSI Server & Local Timer Setup
# ------------------------------
gsi_app = Flask(__name__)
latest_gsi_state = {}
ROUND_FULL_S = 115
BOMB_PLANT_S = 39   
_timer_lock = threading.Lock()
_last_round_phase = None
_last_bomb_planted = None
_timer_start_t = None      
_timer_total_s = None 

def _now():
    return time.monotonic()

def _fmt_mmss(seconds_left):
    if seconds_left is None:
        return "—"
    s = max(0, int(seconds_left))
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

def _remaining_from(start_t, total_s):
    if start_t is None or total_s is None:
        return None
    return max(0.0, total_s - (_now() - start_t))

def get_round_timer_text():
    with _timer_lock:
        start = _timer_start_t
        total = _timer_total_s
    rem = _remaining_from(start, total)
    if rem is not None and rem <= 0:
        with _timer_lock:
            globals()['_timer_start_t'] = None
            globals()['_timer_total_s'] = None
        return "—"
    return _fmt_mmss(rem)

def safe_get(d, path, default=None):
    for key in path:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d != {} else default

@gsi_app.route('/clutch', methods=['POST'])
def gsi_listener():
    global latest_gsi_state
    state = request.json

    # --- Extract data ---
    player = state.get('player', {})
    map_info = state.get('map', {})
    round_info = state.get('round', {})
    weapons = player.get('weapons', {})

    # --- Basic player info ---
    name = player.get('name', 'Unknown')
    team = player.get('team', '?')

    # --- Player state info ---
    st = player.get('state', {})
    hp = st.get('health')
    armor = st.get('armor')
    helmet = st.get('helmet')
    smoked = st.get('smoked')
    money = st.get('money')
    has_kit = st.get('defusekit')
    round_kills = st.get('round_kills')
    round_killhs = st.get('round_killhs')

    # --- Bomb carrier check ---
    has_bomb = any(
        w.get("name") == "weapon_c4"
        for w in weapons.values()
    )

    # --- Match stats ---
    stats = player.get('match_stats', {})
    kills = stats.get('kills')
    assists = stats.get('assists')
    deaths = stats.get('deaths')
    mvps = stats.get('mvps')
    score = stats.get('score')

    # --- Active weapon ---
    active_weapon = None
    for w in weapons.values():
        if w.get('state') == 'active':
            active_weapon = w
            break

    # Build structured GSI snapshot
    latest_gsi_state = {
        "player_name": name,
        "player_team": team,
        "health": hp,
        "armor": armor,
        "helmet": helmet,
        "is_smoked": smoked,
        "money": money,
        "has_defuse_kit": has_kit,
        "kills_this_round": round_kills,
        "headshots_this_round": round_killhs,
        "total_kills_in_match": kills,
        "total_assists_in_match": assists,
        "total_deaths_in_match": deaths,
        "total_MVPs_in_match": mvps,
        "player_score_in_match": score,
        "is_bomb_carrier": has_bomb,
        "active_weapon": {
            "weapon_name": active_weapon.get('name') if active_weapon else None,
            "ammo_in_mag": active_weapon.get('ammo_clip') if active_weapon else None,
            "ammo_in_reserve": active_weapon.get('ammo_reserve') if active_weapon else None
        },
        "all_weapons_in_inventory": [
            {
                "weapon_name": w.get("name"),
                "weapon_type": w.get("type"),
                "weapon_state": w.get("state"),
                "ammo_in_mag": w.get("ammo_clip"),
                "ammo_in_reserve": w.get("ammo_reserve")
            }
            for w in weapons.values()
            if w.get("type") not in ("C4", "Knife")
        ],
        "map_info": {
            "map_name": map_info.get('name'),
            "game_mode": map_info.get('mode'),
            "map_phase": map_info.get('phase'),
            "ct_rounds_won": safe_get(map_info, ['team_ct', 'score']),
            "t_rounds_won": safe_get(map_info, ['team_t', 'score']),
        },
        "round_info": {
            "round_phase": round_info.get('phase'),
            "bomb_status": round_info.get('bomb'),
            "winning_team": round_info.get('win_team')
        }
    }

    # Use data for local timer
    map_mode    = map_info.get('mode')
    round_phase = round_info.get('phase')
    bomb_state  = round_info.get('bomb')
    bomb_planted = (bomb_state == "planted")
    global _last_round_phase, _last_bomb_planted, _timer_start_t, _timer_total_s
    with _timer_lock:
        if map_mode == "competitive":
            if _last_round_phase == "freezetime" and round_phase == "live":
                _timer_start_t = _now()
                _timer_total_s = ROUND_FULL_S
            if (_last_bomb_planted is False or _last_bomb_planted is None) and bomb_planted is True:
                if _timer_start_t is not None:
                    _timer_start_t = _now()
                    _timer_total_s = BOMB_PLANT_S
            if round_phase == "over":
                _timer_start_t = None
                _timer_total_s = None
        else:
            _timer_start_t = None
            _timer_total_s = None
        _last_round_phase = round_phase
        _last_bomb_planted = bomb_planted

    return "ok"

def start_gsi_server():
    gsi_app.run(port=3200, debug=False, use_reloader=False)



# ------------------------
# Screenshot Capture Setup
# ------------------------
def is_cs2_window_visible():
    for w in gw.getWindowsWithTitle("Counter-Strike 2"):
        if w.title and w.left > -10000 and w.top > -10000 and not w.isMinimized:
            print(f"CS2 window detected at ({w.left}, {w.top})")
            return True
    print("CS2 window not found or not visible.")
    return False

def get_screen_size():
    cam = bettercam.create(output_idx=0)
    frame = cam.grab()
    if frame is None:
        raise RuntimeError("Could not detect screen size.")
    return frame.shape[1], frame.shape[0]  

def grab_region(x_frac, y_frac, w_frac, h_frac, screen_w, screen_h):
    left   = int(screen_w * x_frac)
    top    = int(screen_h * y_frac)
    right  = left + int(screen_w * w_frac)
    bottom = top + int(screen_h * h_frac)
    cam = bettercam.create(output_idx=0)
    frame = cam.grab(region=(left, top, right, bottom))
    if frame is None:
        raise RuntimeError(f"Could not grab region ({left},{top},{right},{bottom}).")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def frame_to_model_base64(frame):
    try:
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            print("Failed to encode frame.")
            return None
        b64_str = base64.encodebytes(buffer).decode("utf-8").replace("\n", "")
        return {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64_str}",
            "detail": "high"
        }
    except Exception as e:
        print("Error during screenshot encoding:", e)
        return None

def capture_cs2_images():
    if not is_cs2_window_visible():
        return []
    try:
        # Detect real resolution
        screen_w, screen_h = get_screen_size()

        # Capture regions based on fractions
        radar      = grab_region(0.0, 0.0, 0.2, 0.32, screen_w, screen_h)
        ct_alive   = grab_region(0.2, 0.0, 0.27, 0.12, screen_w, screen_h)
        t_alive    = grab_region(0.53, 0.0, 0.27, 0.12, screen_w, screen_h)

        # Prepare payloads
        payloads = []
        for f in (radar, ct_alive, t_alive):
            p = frame_to_model_base64(f) if f is not None else None
            if p:
                payloads.append(p)

        print(f"[DONE] Prepared {len(payloads)} image payload(s) for model.")
        return payloads
    except Exception as e:
        print("Error during multi-image capture:", e)
        return []



# -----------------------
# Tokens Normalizer Setup
# -----------------------
_ONLY_PUNCT_OR_SPACE = re.compile(r"^[\s\.\,\!\?\:\;\…—–-]+$")

def _pending_pause_for(token: str) -> str:
    tok = token.strip()
    for ch in reversed(tok):
        if ch in ".!?":       
            return ". "
        if ch in ":;…—–-":    
            return ", "
    return ", "

def tts_normalize_for_speech(s: str) -> str:
    if not s:
        return s

    # Remove zero-width & bidi marks that can cause artifacts
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]", "", s)

    # Strip markdown emphasis / stray asterisks
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = s.replace("*", "")

    # Normalize smart quotes to ASCII 
    s = s.replace("\u2018", "'").replace("\u2019", "'")   
    s = s.replace("\u201C", '"').replace("\u201D", '"')  

    # Keep . ! ? as strong pauses; convert others to short pauses 
    s = re.sub(r"[—–]+", ",", s)          
    s = re.sub(r"[;:]+", ",", s)          
    s = re.sub(r"\.{3,}|[…]+", ",", s)    

    # Fix stray space before punctuation & keep decimals intact
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)

    # Tidy commas
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*", ", ", s)
    s = re.sub(r"(,\s*){2,}", ", ", s)

    return s.strip()



# -----------------------
# STT & TTS Workers Setup
# -----------------------
class STTWorker(QThread):
    stt_partial = pyqtSignal(str) 
    stt_final = pyqtSignal(str) 
    ai_stream_started = pyqtSignal()    
    ai_stream_token = pyqtSignal(str)   
    ai_stream_done = pyqtSignal()       
    stt_ready = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.end_of_sentence_detection_pause = 0.5
        self.unknown_sentence_detection_pause = 1.0
        self.mid_sentence_detection_pause = 2.5
        self.full_sentences = []
        self.displayed_text = ""
        self.prev_text = ""
        self.recorder = None
        self.current_stream = None
        self.use_piper = False

        # Initialize the TTS engine depending on remaining VRAM
        if torch.cuda.is_available():
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                total_gb = total_mem / (1024 ** 3)
                print(f"[VRAM] Total: {total_gb:.2f} GB")
                self.use_piper = total_gb < 12.0
            except Exception as e:
                print("VRAM check failed, defaulting to Piper:", e)
        if self.use_piper:
            print("🔊 Using Piper TTS (low-VRAM mode)")
            piper_bin   = os.path.join(base_dir, "models", "Piper", "piper", "piper.exe")
            model_file  = os.path.join(base_dir, "models", "Piper", "en_US-norman-medium.onnx")
            config_file = os.path.join(base_dir, "models", "Piper", "en_US-norman-medium.onnx.json")
            voice  = PiperVoice(model_file=model_file, config_file=config_file)
            engine = PiperEngine(piper_path=piper_bin, voice=voice, debug=False)
        else:
            print("🔊 Using Coqui XTTS (high-VRAM mode)")
            model_path = os.path.join(base_dir, "models", "Coqui")
            engine = CoquiEngine(
                specific_model=model_path,
                voice="Damien Black",
                speed=1.1,
                thread_count=5,
                temperature=0.95,
                repetition_penalty=5.0,
                top_k=30,
                top_p=0.92,
                device="cuda"
            )
        self.tts_stream = TextToAudioStream(engine=engine)

        # Warm up TTS engine with a silent dummy token
        try:
            self.tts_stream.feed(" ")
            self.tts_stream.play_async()
            time.sleep(0.05)  
            self.tts_stream.stop()
            print("TTS engine warmed up.")
        except Exception as e:
            print("TTS engine warm-up failed:", e)

    def on_wakeword_detected(self):
        # Stop TTS if playing
        try:
            if self.tts_stream.is_playing():
                self.tts_stream.stop()
        except Exception as e:
            print("Error stopping TTS:", e)

        # Close LLM stream if active
        try:
            if self.current_stream:
                self.current_stream.close()
                self.current_stream = None
        except Exception as e:
            print("Error closing current stream:", e)

    def preprocess_text(self, text):
        text = text.lstrip()
        if text.startswith("..."):
            text = text[3:]
        text = text.lstrip()
        if text:
            text = text[0].upper() + text[1:]
        return text

    def text_detected(self, text):
        text = self.preprocess_text(text)
        sentence_end_marks = ['.', '!', '?', '。']

        if text.endswith("..."):
            self.recorder.post_speech_silence_duration = self.mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and self.prev_text and self.prev_text[-1] in sentence_end_marks:
            self.recorder.post_speech_silence_duration = self.end_of_sentence_detection_pause
        else:
            self.recorder.post_speech_silence_duration = self.unknown_sentence_detection_pause

        self.prev_text = text

        # Build output
        styled = ""
        for i, sentence in enumerate(self.full_sentences):
            styled += sentence + " "

        if text:
            styled += f"<b>{text}</b>"

        # Send real-time strings for the UI panel
        self.stt_partial.emit(styled.replace("<b>", "").replace("</b>", ""))

    def process_text(self, text):
        self.recorder.post_speech_silence_duration = self.unknown_sentence_detection_pause
        text = self.preprocess_text(text).rstrip()
        if text.endswith("..."):
            text = text[:-2]
        if not text:
            return

        self.full_sentences.append(text)
        self.prev_text = ""
        self.text_detected("")

        # --- Create prompts for the brain ---

        # 1. Store and show final user input
        self.user_input = text
        self.stt_final.emit(self.user_input)
        self.ai_stream_started.emit()

        # 2. Store latest GSI data
        latest_gsi = latest_gsi_state.copy() if latest_gsi_state else {}

        # 3. Capture CS2 screenshot (returns model-ready payload or None)
        image_payloads = capture_cs2_images()

        # 4. Construct the dynamic prompt for model
        system_prompt = (
            "You are Jarvis, a super chill and witty Counter-Strike 2 legendary coach.\n\n"
            "- For urgent, in-round situations: give a decisive, real-time main plan to execute in 3-5s, and sometimes a tiny backup if it fails. Must sound natural.\n"
            "- For general/non-urgent questions: answer naturally but stay brief and actionable.\n"
            "- Max 3 short-medium sentences.\n"
            "- Speak like a Tier-1 IGL or ex-pro, not a generic AI.\n"
            "- Prioritize: round win > survival > economy.\n"
            "- Use only true CS2 knowledge. Never invent callouts, lineups, or mechanics.\n"
            "- Use full words, no slang or abbreviations.\n"
            "- Utilize all given information.\n"
            "Every response must be instantly usable and realistic."
        )

        image_analysis_prompt = """
        You will also receive up to three labeled images in this order:
        1. **Minimap** - The radar circle with the user in the center. Right beneath the radar circle is a text saying **EXACTLY** where the user is in the map. Use both the radar and text to *PRECISELY* determine user's location relative to surroundings. Optionally, if visible, detect red circular dots (enemy positions) and use them to infer enemy locations relative to the user.
        2. **CT Players Alive** - Shows the number of living CT players (big blue number). If avatars are shown instead, count bright avatars as alive and greyed-out avatars as dead.
        3. **T Players Alive** - Shows the number of living T players (big yellow number). If avatars are shown instead, count bright avatars as alive and greyed-out avatars as dead.
        Utilize the information from these images to support your tactical reasoning and make your advice as precise as possible.
        """

        if image_payloads:
            system_prompt += "\n" + image_analysis_prompt

        # 5. Construct the dynamic content for user
        round_timer_txt = get_round_timer_text()

        if round_timer_txt and round_timer_txt != "—":
            timer_str = f"\n\n**Time Left in Round:** {round_timer_txt}"
        else:
            timer_str = ""

        self.prompt = f"""
            **User's Input:**
            \"{self.user_input}\"

            **Game State:**
            {json.dumps(latest_gsi, indent=2)}
            {timer_str}
            """.strip()

        user_content = [
            { "type": "input_text", "text": self.prompt }
        ]

        if image_payloads:
            labels = ["Minimap", "CT Players Alive", "T Players Alive" ]
            for label, img_payload in zip(labels, image_payloads):
                user_content.append({ "type": "input_text", "text": f"[{label}]" })
                user_content.append(img_payload)

        # --- Run the brain, feed response from model chunk by chunk into the TTS, and make it speak ASAP ---

        # 1. Prepare the message
        message = [
            { "role": "developer", "content": system_prompt },
            { "role": "user", "content": user_content }
        ]

        # 2. Clear any leftover audio before new response
        try:
            self.tts_stream.stop()
        except Exception:
            pass
        try:
            eng = getattr(self.tts_stream, "engine", None)
            q = getattr(eng, "queue", None)
            if q is not None and hasattr(q, "queue"):
                with q.mutex:
                    q.queue.clear()
        except Exception:
            pass

        # 3. Call the brain
        print("Sending data to model...")
        stream = call_openai(message)
        if not stream:
            print("No response from the model.")
            return
        self.current_stream = stream
        
        # 4. Feed the response stream into the correct TTS
        print(f"Receiving chunks from model...")
        if self.use_piper:
            started = False
            buf = []
            buf_chars = 0
            last_flush = time.monotonic()
            pending_pause = "" 
            FIRST_MAX_CHARS = 140
            FIRST_MAX_WAIT  = 1.0
            NEXT_MAX_CHARS  = 90
            NEXT_MAX_WAIT   = 0.8
            PIPER_PUNCT = (".", "!", "?", "…", ":", ";", "\n")
            def thresholds():
                return (FIRST_MAX_CHARS, FIRST_MAX_WAIT) if not started else (NEXT_MAX_CHARS, NEXT_MAX_WAIT)
            def flush_buffer():
                nonlocal buf, buf_chars, started, last_flush, pending_pause 
                if not buf:
                    last_flush = time.monotonic()
                    return
                text = "".join(buf).strip()
                buf.clear()
                buf_chars = 0
                if not text:
                    last_flush = time.monotonic()
                    return
                tts_text = tts_normalize_for_speech(text)
                if _ONLY_PUNCT_OR_SPACE.match(tts_text):
                    pending_pause = _pending_pause_for(tts_text)
                    last_flush = time.monotonic()
                    return
                if re.search(r'([.!?])\s*(["\']?)\s*$', tts_text):
                    tts_text = re.sub(r'\s*$', '', tts_text) + "\n"
                if pending_pause:
                    self.tts_stream.feed(pending_pause)
                    pending_pause = ""
                    if not started:
                        time.sleep(0.15)
                        self.tts_stream.play_async()
                        started = True
                self.tts_stream.feed(tts_text)
                if not started:
                    time.sleep(0.15)
                    self.tts_stream.play_async()
                    started = True
                last_flush = time.monotonic()
            try:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        token = event.delta
                        self.ai_stream_token.emit(token)
                        buf.append(token)
                        buf_chars += len(token)
                        max_chars, max_wait = thresholds()
                        if (
                            (token and token[-1] in PIPER_PUNCT) or
                            (buf_chars >= max_chars) or
                            ((time.monotonic() - last_flush) > max_wait)
                        ):
                            flush_buffer()
                flush_buffer()
                self.ai_stream_done.emit()
            except Exception:
                print("Error during stream playback (Piper):")
                traceback.print_exc()
        else:
            started = False
            pending_pause = "" 
            try:
                feed = self.tts_stream.feed
                play = self.tts_stream.play_async
                for event in stream:
                    if event.type == "response.output_text.delta":
                        token = event.delta
                        self.ai_stream_token.emit(token)
                        if token.strip() == "":
                            continue
                        stripped = token.strip()
                        if re.fullmatch(r"[\.\!\?\:\;\…—–-]+", stripped):
                            pending_pause = _pending_pause_for(stripped)
                            continue
                        out = tts_normalize_for_speech(token)
                        if re.search(r'([.!?])\s*(["\']?)\s*$', out):
                            out = re.sub(r'\s*$', '', out) + "\n"
                        if pending_pause:
                            feed(pending_pause)
                            pending_pause = ""
                            if not started:
                                play()
                                started = True
                        feed(out)
                        if not started:
                            play()
                            started = True
                self.ai_stream_done.emit()
            except Exception:
                print("Error during stream playback (Coqui):")
                traceback.print_exc()

    def run(self):
        recorder_config = {
            'wakeword_backend': 'openwakeword',
            'wake_words': "hey_jarvis",
            'on_wakeword_detected': self.on_wakeword_detected,
            'model': 'models/tiny.en',
            'realtime_model_type': 'models/tiny.en',
            'language': 'en',
            'early_transcription_on_silence': 0.2,
            'post_speech_silence_duration': self.unknown_sentence_detection_pause,
            'min_length_of_recording': 1.0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.02,
            'on_realtime_transcription_update': self.text_detected,
            'silero_deactivity_detection': True,
            'beam_size_realtime': 4,
            'silero_use_onnx': True,
            'faster_whisper_vad_filter': False,
            'no_log_file': True,
        }

        self.recorder = AudioToTextRecorder(**recorder_config)

        try:
            self.stt_ready.emit()
        except Exception:
            pass

        while True:
            self.recorder.text(self.process_text)



# -------------------
# Clutch App UI Setup
# -------------------
class ClutchWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setVisible(False)
        self.setMinimumSize(960, 560)
        self.setStyleSheet("""
            QWidget { background: #0B0B0C; color: #EAEAEA; font-family: Inter, "Segoe UI", Roboto, "SF Pro Text", Arial; }
            QLabel#Title {
                font-size: 26px; font-weight: 800; letter-spacing: 1.2px; color: #FFFFFF;
            }
            QLabel#Badge {
                padding: 6px 12px; border-radius: 10px; border: 1px solid #2B2B2F;
                color: #DADADA; background: #111113; font-weight: 600;
            }
            QLabel#Section {
                font-size: 12px; font-weight: 700; letter-spacing: .6px; color: #CFCFD4;
            }
            QLabel#Mono {
                font-family: "Cascadia Code", Menlo, Consolas, monospace;
                font-size: 14px; line-height: 1.5em; color: #E6E6E6;
            }
            QFrame#Card { background: #121214; border: 1px solid #1F1F22; border-radius: 16px; }

            /* removed inner black rectangles */
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #2A2A2E; border-radius: 6px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background: #343439; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        header = QHBoxLayout()
        header.setContentsMargins(12, 14, 12, 10)
        header.setSpacing(12)
        self.title = QLabel("CLUTCH")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignHCenter)
        header.addStretch(1)
        header.addWidget(self.title, 0, Qt.AlignHCenter)
        header.addStretch(1)
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(12, 0, 12, 8)
        badge_col = QVBoxLayout()
        badge_col.setSpacing(40) 
        badge_col.setAlignment(Qt.AlignHCenter)
        self.badge_tts = QLabel("TTS engine: —")
        self.badge_tts.setObjectName("Badge")
        self.badge_tts.setAlignment(Qt.AlignHCenter)
        badge_col.addWidget(self.badge_tts)
        self.badge_timer = QLabel("⏱ —")
        self.badge_timer.setObjectName("Badge")
        self.badge_timer.setAlignment(Qt.AlignHCenter)
        badge_col.addWidget(self.badge_timer)
        badge_row.addLayout(badge_col)
        self._timer_updater = QTimer(self)
        self._timer_updater.timeout.connect(self._refresh_round_timer_badge)
        self._timer_updater.start(100)
        body = QHBoxLayout()
        body.setSpacing(16)
        def make_card():
            card = QFrame()
            card.setObjectName("Card")
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(28)
            shadow.setOffset(0, 6)
            shadow.setColor(Qt.black)
            card.setGraphicsEffect(shadow)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(16, 14, 16, 16)
            lay.setSpacing(10)
            return card, lay
        self.user_card, u = make_card()
        u_title = QLabel("USER'S INPUT")
        u_title.setObjectName("Section")
        u_title.setAlignment(Qt.AlignHCenter)
        u_title.setStyleSheet("background-color: transparent;")
        u_scroll = QScrollArea()
        u_scroll.setWidgetResizable(True)
        self.user_inner = QWidget()
        self.user_inner.setStyleSheet("""
            background-color: #000000;
            border-radius: 12px;
        """)
        self.user_inner_lay = QVBoxLayout(self.user_inner)
        self.user_inner_lay.setContentsMargins(14, 14, 14, 14)
        self.user_inner_lay.setSpacing(0)
        self.stt_label = QLabel('Say "Hey Jarvis" to start…')
        self.stt_label.setObjectName("Mono")
        self.stt_label.setWordWrap(True)
        self.stt_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.user_inner_lay.addWidget(self.stt_label)
        u_scroll.setWidget(self.user_inner)
        u.addWidget(u_title)
        u.addWidget(u_scroll)
        self.ai_card, a = make_card()
        a_title = QLabel("MODEL'S RESPONSE")
        a_title.setObjectName("Section")
        a_title.setAlignment(Qt.AlignHCenter)
        a_title.setStyleSheet("background-color: transparent;")
        a_scroll = QScrollArea()
        a_scroll.setWidgetResizable(True)
        self.ai_inner = QWidget()
        self.ai_inner.setStyleSheet("""
            background-color: #000000;
            border-radius: 12px;
        """)
        self.ai_inner_lay = QVBoxLayout(self.ai_inner)
        self.ai_inner_lay.setContentsMargins(14, 14, 14, 14)
        self.ai_inner_lay.setSpacing(0)
        self.ai_label = QLabel("Waiting for a prompt…")
        self.ai_label.setObjectName("Mono")
        self.ai_label.setWordWrap(True)
        self.ai_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.ai_inner_lay.addWidget(self.ai_label)
        a_scroll.setWidget(self.ai_inner)
        a.addWidget(a_title)
        a.addWidget(a_scroll)
        body.addWidget(self.user_card, 1)
        body.addWidget(self.ai_card, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 18)
        root.setSpacing(8)
        root.addLayout(header)
        root.addLayout(badge_row)
        root.addLayout(body)
        self.thread = STTWorker()
        self.thread.stt_partial.connect(self.on_stt_partial)
        self.thread.stt_final.connect(self.on_stt_final)
        self.thread.ai_stream_started.connect(self.on_ai_stream_started)
        self.thread.ai_stream_token.connect(self.on_ai_stream_token)
        self.thread.ai_stream_done.connect(self.on_ai_stream_done)
        self.thread.stt_ready.connect(self.on_stt_ready)
        try:
            self.badge_tts.setText(f"TTS engine: {'Piper' if self.thread.use_piper else 'Coqui XTTS'}")
        except Exception:
            pass
        self.thread.start()
        self.latest_user_text = ""
        self._user_scroll = u_scroll
        self._ai_scroll = a_scroll

    def _refresh_round_timer_badge(self):
        try:
            txt = get_round_timer_text()
            self.badge_timer.setText(f"⏱ {txt}")
        except Exception:
            self.badge_timer.setText("⏱ —")

    def on_stt_ready(self):
        self.setWindowTitle("CLUTCH")
        self.show()

    def _scroll_to_bottom(self, scroll):
        bar = scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _last_segment(text: str) -> str:
        import re
        parts = re.split(r'[.!?\n\u3002]+', text)
        for seg in reversed(parts):
            seg = seg.strip()
            if seg:
                return seg
        return text.strip()

    def on_tts_mode(self, mode_name: str):
        self.badge_tts.setText(f"TTS engine: {mode_name}")

    def on_stt_partial(self, text: str):
        self.latest_user_text = self._last_segment(text)
        self.stt_label.setText(self.latest_user_text)
        self._scroll_to_bottom(self._user_scroll)

    def on_stt_final(self, text: str):
        self.latest_user_text = text.strip() or "—"
        self.stt_label.setText(self.latest_user_text)
        self._scroll_to_bottom(self._user_scroll)

    def on_ai_stream_started(self):
        self.ai_label.setText("")
        self._scroll_to_bottom(self._ai_scroll)

    def on_ai_stream_token(self, token: str):
        self.ai_label.setText(self.ai_label.text() + token)
        self._scroll_to_bottom(self._ai_scroll)

    def on_ai_stream_done(self):
        pass

    def closeEvent(self, event):
        try:
            if hasattr(self, "thread") and self.thread.isRunning():
                try:
                    if getattr(self.thread, "recorder", None):
                        if hasattr(self.thread.recorder, "stop"):
                            self.thread.recorder.stop()
                        if hasattr(self.thread.recorder, "shutdown"):
                            self.thread.recorder.shutdown()
                        if hasattr(self.thread.recorder, "close"):
                            self.thread.recorder.close()
                except Exception:
                    pass
                self.thread.terminate()
            if hasattr(self.thread, "tts_stream"):
                try:
                    self.thread.tts_stream.stop()
                except Exception:
                    pass
        except Exception:
            pass
        os.kill(os.getpid(), signal.SIGINT)



# Function to start the UI
def start_ui():
    app = QApplication(sys.argv)
    window = ClutchWindow()
    sys.exit(app.exec_())

# Main function to start everything
def main():
    # 1. Set up GSI config file
    setup_gsi_cfg()

    # 2. Warm up model in the background
    warmup_openai()

    # 3. Start application thread (non-daemon)
    ui_thread = threading.Thread(target=start_ui, daemon=False)
    ui_thread.start()

    # 4. Start GSI server in background thread (daemon)
    gsi_thread = threading.Thread(target=start_gsi_server, daemon=True)
    gsi_thread.start()

    # 5. Keep main thread alive
    while True:
        time.sleep(1)

# Function to warm up OpenAI model
def warmup_openai():
    def _warm():
        try:
            _ = openai_client.responses.create(
                model="gpt-5-chat-latest",
                input=[{
                    "role": "user",
                    "content": "Say hi and nothing else."
                }],
                temperature=0.0,
                stream=False
            )
            print("OpenAI warmed up.")
        except Exception as e:
            print("OpenAI warm-up failed:", e)
    threading.Thread(target=_warm, daemon=True).start()

if __name__ == "__main__":
    main()
