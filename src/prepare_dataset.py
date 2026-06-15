import os
import json

CLEAN_DIR = "data/raw"
NOISY_DIR = "data/noisy"
OUTPUT_FILE = "data/processed/dataset.json"

COMMANDS = [
    "Check blood pressure", "Measure heart rate",
    "Record temperature", "Check oxygen saturation",
    "Monitor pulse", "Check respiratory rate",
    "Start IV drip", "Stop IV drip",
    "Increase oxygen flow", "Decrease oxygen flow",
    "Start ventilator", "Stop ventilator",
    "Silence alarm", "Reset monitor",
    "Patient is stable", "Patient is critical",
    "Patient needs medication", "Call the nurse",
    "Request doctor", "Paracetamol",
    "Amoxicillin", "Ibuprofen", "Morphine",
    "Insulin", "Metformin", "Begin procedure",
    "End procedure", "Prepare operating room",
    "Request blood sample", "Start ECG"
]

def extract_command_from_filename(filename):
    """Extract the command text from the filename."""
    name = filename.replace('.wav', '')
    # Remove SNR suffix if present
    for snr in ['_snr20db', '_snr10db', '_snr0db']:
        name = name.replace(snr, '')
    # Remove command index prefix (e.g., cmd01_)
    parts = name.split('_', 1)
    if len(parts) > 1:
        command_part = parts[1].replace('_', ' ')
        # Match to closest command
        for cmd in COMMANDS:
            if cmd.lower() == command_part.lower():
                return cmd
    return None

def build_dataset():
    print("Building dataset manifest...")
    entries = []

    # Clean audio
    for speaker in os.listdir(CLEAN_DIR):
        speaker_path = os.path.join(CLEAN_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for wav_file in os.listdir(speaker_path):
            if not wav_file.endswith('.wav'):
                continue
            command = extract_command_from_filename(wav_file)
            if command:
                entries.append({
                    "audio_path": os.path.join(CLEAN_DIR, speaker, wav_file),
                    "transcript": command,
                    "speaker": speaker,
                    "noise_level": "clean"
                })

    # Noisy audio
    for speaker in os.listdir(NOISY_DIR):
        speaker_path = os.path.join(NOISY_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for snr_dir in os.listdir(speaker_path):
            snr_path = os.path.join(speaker_path, snr_dir)
            for wav_file in os.listdir(snr_path):
                if not wav_file.endswith('.wav'):
                    continue
                command = extract_command_from_filename(wav_file)
                if command:
                    entries.append({
                        "audio_path": os.path.join(snr_path, wav_file),
                        "transcript": command,
                        "speaker": speaker,
                        "noise_level": snr_dir
                    })

    # Save manifest
    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"✅ Dataset manifest created: {OUTPUT_FILE}")
    print(f"   Total entries: {len(entries)}")

if __name__ == "__main__":
    build_dataset()