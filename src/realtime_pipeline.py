import torch
import whisper
import numpy as np
import sounddevice as sd
import librosa
import os
import noisereduce as nr


# ── Settings ──────────────────────────────────────────
MODEL_SIZE   = "base"
MODEL_PATH   = "models/shaavox_med.pt"
SAMPLE_RATE  = 16000
DURATION     = 3       # seconds to record per command
# ──────────────────────────────────────────────────────

COMMANDS = [
    "check blood pressure", "measure heart rate",
    "record temperature", "check oxygen saturation",
    "monitor pulse", "check respiratory rate",
    "start iv drip", "stop iv drip",
    "increase oxygen flow", "decrease oxygen flow",
    "start ventilator", "stop ventilator",
    "silence alarm", "reset monitor",
    "patient is stable", "patient is critical",
    "patient needs medication", "call the nurse",
    "request doctor", "paracetamol",
    "amoxicillin", "ibuprofen", "morphine",
    "insulin", "metformin", "begin procedure",
    "end procedure", "prepare operating room",
    "request blood sample", "start ecg",
]


def match_command(transcript):
    """Match transcribed text to closest known command."""
    transcript = transcript.lower().strip()
    # Exact match first
    if transcript in COMMANDS:
        return transcript, True
    # Partial match
    for cmd in COMMANDS:
        if cmd in transcript or transcript in cmd:
            return cmd, True
    # Word overlap
    words = set(transcript.split())
    best_cmd, best_score = None, 0
    for cmd in COMMANDS:
        overlap = len(words & set(cmd.split()))
        if overlap > best_score:
            best_score = overlap
            best_cmd = cmd
    if best_score >= 1:
        return best_cmd, False
    return None, False


def record_audio():
    """Record audio from microphone."""
    print("🔴 Recording... speak now!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✅ Recording done!")
    return audio.flatten()


def transcribe(model, audio):
    """Transcribe audio using ShaaVox-Med model."""
    audio = whisper.pad_or_trim(audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text.strip()

def reduce_noise(audio):
    """Apply noise reduction before transcription."""
    # Use first 0.5 seconds as noise profile
    noise_sample = audio[:SAMPLE_RATE // 2]
    reduced = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        y_noise=noise_sample,
        prop_decrease=0.75
    )
    return reduced


def run_pipeline():
    print("=" * 55)
    print("   ShaaVox-Med Real-Time Pipeline")
    print("=" * 55)

    # Load model
    print("\n⏳ Loading ShaaVox-Med model...")
    model = whisper.load_model(MODEL_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("✅ Model ready!\n")

    print("📋 System ready. Speak one of the 30 medical commands.")
    print("   Type 'quit' to exit.\n")

    while True:
        user_input = input("Press ENTER to record (or type 'quit' to exit): ").strip().lower()
        if user_input == 'quit':
            print("👋 Exiting ShaaVox-Med.")
            break

        # Record
        audio = record_audio()

        # Reduce noise 
        audio = reduce_noise(audio)

        # Transcribe
        print("🧠 Transcribing...")
        transcript = transcribe(model, audio)
        print(f"📝 Heard:    '{transcript}'")

        # Match to command
        matched_cmd, exact = match_command(transcript)
        if matched_cmd:
            status = "✅ EXACT MATCH" if exact else "🔶 CLOSEST MATCH"
            print(f"{status}: '{matched_cmd.upper()}'")
        else:
            print("❌ No matching command found.")

        print("-" * 55)


if __name__ == "__main__":
    run_pipeline()