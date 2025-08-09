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
import filecmp
import time
import base64
import cv2
import bettercam
import pygetwindow as gw
import traceback
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, CoquiEngine
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



# ----------------
# GSI Server Setup
# ----------------
gsi_app = Flask(__name__)
latest_gsi_state = {}

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
    countdowns = state.get('phase_countdowns', {})
    weapons = player.get('weapons', {})

    # --- Basic player info ---
    name = player.get('name', 'Unknown')
    team = player.get('team', '?')
    steamid = player.get('steamid')

    # --- Player state info ---
    st = player.get('state', {})
    hp = st.get('health')
    armor = st.get('armor')
    helmet = st.get('helmet')
    flashed = st.get('flashed')
    smoked = st.get('smoked')
    burning = st.get('burning')
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
        "steam_id": steamid,
        "health": hp,
        "armor": armor,
        "helmet": helmet,
        "is_flashed": flashed,
        "is_smoked": smoked,
        "is_burning": burning,
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
        ],
        "map_info": {
            "map_name": map_info.get('name'),
            "game_mode": map_info.get('mode'),
            "map_phase": map_info.get('phase'),
            "ct_rounds_won": safe_get(map_info, ['team_ct', 'score']),
            "t_rounds_won": safe_get(map_info, ['team_t', 'score']),
            "ct_consecutive_round_losses": safe_get(map_info, ['team_ct', 'consecutive_round_losses']),
            "t_consecutive_round_losses": safe_get(map_info, ['team_t', 'consecutive_round_losses'])
        },
        "round_info": {
            "round_phase": round_info.get('phase'),
            "bomb_status": round_info.get('bomb'),
            "winning_team": round_info.get('win_team')
        }
    }

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
        overall    = grab_region(0.0, 0.0, 1.0, 1.0, screen_w, screen_h)
        radar      = grab_region(0.0, 0.0, 0.2, 0.3, screen_w, screen_h)
        ct_alive   = grab_region(0.2, 0.0, 0.27, 0.12, screen_w, screen_h)
        t_alive    = grab_region(0.53, 0.0, 0.27, 0.12, screen_w, screen_h)

        # Prepare payloads
        payloads = []
        for f in (overall, radar, ct_alive, t_alive):
            p = frame_to_model_base64(f) if f is not None else None
            if p:
                payloads.append(p)

        print(f"[DONE] Prepared {len(payloads)} image payload(s) for model.")
        return payloads
    except Exception as e:
        print("Error during multi-image capture:", e)
        return []



# -----------------------
# STT & TTS Workers Setup
# -----------------------
class STTWorker(QThread):
    update = pyqtSignal(str)

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

        # Initialize the TTS engine, too
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
            self.tts_stream.feed("...")
            self.tts_stream.play_async()
            time.sleep(1.0)  
            self.tts_stream.stop()
            print("Warmed up Coqui TTS...")
        except Exception as e:
            print("TTS warm-up failed:", e)

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

        self.update.emit(styled)

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

        # 1. Store final user input
        self.user_input = text

        # 2. Store latest GSI data
        latest_gsi = latest_gsi_state.copy() if latest_gsi_state else {}

        # 3. Capture CS2 screenshot (returns model-ready payload or None)
        image_payloads = capture_cs2_images()

        # 4. Construct the dynamic prompt for model
        system_prompt = (
            "You are Jarvis, a top-tier, friendly, and witty Counter-Strike 2 coach who can also chat.\n\n"
            "- Give actionable, real-time coaching using true CS2 knowledge, in max 3 sentences (10-15 seconds spoken max).\n"
            "- Speak like a sharp Tier-1 IGL or ex-pro teammate — never a generic AI.\n"
            "- Prioritize round win > player survival > economy impact.\n"
            "- Use full words — never abbreviations or slang.\n"
            "- Never invent callouts, lineups, or mechanics.\n"
            "- Utilize all given information.\n"
            "Every response must be instantly usable and realistic."
        )

        image_analysis_prompt = """
        You will also receive up to four labeled images in this order:
        1. **Full POV** - The user's current full in-game view. Use this to determine the exact location of the user on the map (you will know the map name from GSI data). Look for map-specific landmarks, textures, and surroundings to pinpoint the user's position.
        2. **Minimap** - The radar circle. The user is at radar center. Use this to refine the user's location by understanding their position relative to surroundings. Optionally, if visible, detect red circular dots (enemy positions) and use them to infer enemy locations relative to the user.
        3. **CT Players Alive** - Shows the number of living CT players (big blue number). If avatars are shown instead, count bright avatars as alive and greyed-out avatars as dead.
        4. **T Players Alive** - Shows the number of living T players (big yellow number). If avatars are shown instead, count bright avatars as alive and greyed-out avatars as dead.
        Use the information from these images to support your tactical reasoning and make your advice as precise as possible.
        """

        if image_payloads:
            system_prompt += "\n" + image_analysis_prompt

        # 5. Construct the dynamic content for user
        self.prompt = f"""
            **User's Input:**
            \"{self.user_input}\"

            **Game State:**
            {json.dumps(latest_gsi, indent=2)}
            """.strip()

        user_content = [
            { "type": "input_text", "text": self.prompt }
        ]

        if image_payloads:
            labels = ["Full POV", "Minimap", "CT Players Alive", "T Players Alive" ]
            for label, img_payload in zip(labels, image_payloads):
                user_content.append({ "type": "input_text", "text": f"[{label}]" })
                user_content.append(img_payload)

        # --- Run the brain, feed response from model chunk by chunk into the TTS, and make it speak ASAP ---

        # 1. Prepare the message
        message = [
            { "role": "developer", "content": system_prompt },
            { "role": "user", "content": user_content }
        ]

        # 2. Call the brain
        print("Sending data to model...")
        stream = call_openai(message)
        if not stream:
            print("No response from the model.")
            return
        self.current_stream = stream
        
        # 3. Feed the response stream into TTS
        print(f"Receiving chunks from model...")
        started = False
        try:
            for event in stream:
                if event.type == "response.output_text.delta":
                    token = event.delta
                    self.tts_stream.feed(token)
                    if not started:
                        self.tts_stream.play_async()
                        started = True
        except Exception as e:
            print("Error during stream playback:")
            traceback.print_exc()

    def run(self):
        recorder_config = {
            'wakeword_backend': 'openwakeword',
            'wake_words': "hey_jarvis",
            'on_wakeword_detected': self.on_wakeword_detected,
            'model': 'models/tiny.en',
            'realtime_model_type': 'models/tiny.en',
            'language': 'en',
            'silero_sensitivity': 0.2,
            'webrtc_sensitivity': 2,
            'early_transcription_on_silence': 0.2,
            'post_speech_silence_duration': self.unknown_sentence_detection_pause,
            'min_length_of_recording': 1.0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.02,
            'on_realtime_transcription_update': self.text_detected,
            'silero_deactivity_detection': True,
            'beam_size': 3,
            'beam_size_realtime': 3,
            'silero_use_onnx': True,
            'faster_whisper_vad_filter': False,
            'no_log_file': True,
        }

        self.recorder = AudioToTextRecorder(**recorder_config)

        while True:
            self.recorder.text(self.process_text)



# -------------------
# Clutch App UI Setup
# -------------------
class ClutchWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLUTCH Voice Assistant")
        self.setGeometry(200, 200, 600, 120)

        self.label = QLabel("🎙️ Say 'Hey Jarvis' and start talking...", self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("font-size: 18px; padding: 10px; color: white;")
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.setStyleSheet("background-color: #121212;")

        self.thread = STTWorker()
        self.thread.update.connect(self.update_text)
        self.thread.start()

    def update_text(self, text):
        self.label.setText(text)



# Function to start the UI
def start_ui():
    app = QApplication(sys.argv)
    window = ClutchWindow()
    window.show()
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
