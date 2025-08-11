# google_tts_test.py
import os
import pyaudio
from google.cloud import texttospeech
from google.oauth2 import service_account

base_dir = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(base_dir, "google_tts_key.json")
assert os.path.isfile(KEY_PATH), f"Missing key: {KEY_PATH}"

creds = service_account.Credentials.from_service_account_file(KEY_PATH)
client = texttospeech.TextToSpeechClient(credentials=creds)

# Don’t set streaming_audio_config; default is LINEAR16/PCM for Chirp 3 HD.
# (Streaming = Chirp 3 HD only; default output format is LINEAR16.) 
# We’ll assume 24 kHz for playback.
SAMPLE_RATE = 24000

streaming_config = texttospeech.StreamingSynthesizeConfig(
    voice=texttospeech.VoiceSelectionParams(
        name="en-US-Chirp3-HD-Algieba",
        language_code="en-US",
    )
)

config_request = texttospeech.StreamingSynthesizeRequest(
    streaming_config=streaming_config
)

text_chunks = [
    "Hey! This is a real-time streaming test. ",
    "Audio should start while text is still sending. ",
    "Latency should feel snappy.",
]

def request_generator():
    yield config_request
    for t in text_chunks:
        yield texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(text=t)
        )

# Raw PCM playback
p = pyaudio.PyAudio()
out = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    output=True,
    frames_per_buffer=2048,
)

try:
    for i, resp in enumerate(client.streaming_synthesize(request_generator()), 1):
        if resp.audio_content:
            out.write(resp.audio_content)  # write raw PCM frames
        print(f"Chunk {i}: {len(resp.audio_content)} bytes")
finally:
    out.stop_stream()
    out.close()
    p.terminate()

print("Done streaming.")
