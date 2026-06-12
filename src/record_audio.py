import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import time
import librosa
import soundfile as sf

# Settings
SAMPLE_RATE = 16000
DURATION = 3
OUTPUT_DIR = "data/raw"

COMMANDS = [
    "Check blood pressure",
    "Measure heart rate",
    "Record temperature",
    "Check oxygen saturation",
    "Monitor pulse",
    "Check respiratory rate",
    "Start IV drip",
    "Stop IV drip",
    "Increase oxygen flow",
    "Decrease oxygen flow",
    "Start ventilator",
    "Stop ventilator",
    "Silence alarm",
    "Reset monitor",
    "Patient is stable",
    "Patient is critical",
    "Patient needs medication",
    "Call the nurse",
    "Request doctor",
    "Paracetamol",
    "Amoxicillin",
    "Ibuprofen",
    "Morphine",
    "Insulin",
    "Metformin",
    "Begin procedure",
    "End procedure",
    "Prepare operating room",
    "Request blood sample",
    "Start ECG",
]

def record_command(command):
    print(f"\nSay: '{command}'")
    print("Recording in 3...", end=" ", flush=True)
    time.sleep(1)
    print("2...", end=" ", flush=True)
    time.sleep(1)
    print("1...", end=" ", flush=True)
    time.sleep(1)
    print("RECORDING NOW!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    print("Recording finished!")
    return audio

def playback_audio(audio):
    print("Playing back your recording...")
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Playback finished!")

def save_command(audio, command, speaker_id, command_idx):
    speaker_dir = os.path.join(OUTPUT_DIR, f"speaker_{speaker_id}")
    
    try:
        os.makedirs(speaker_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return None
    
    filename = f"cmd{command_idx:02d}_{command.replace(' ', '_')}.wav"
    filepath = os.path.join(speaker_dir, filename)
    
    try:
        wav.write(filepath, SAMPLE_RATE, audio)
        if os.path.exists(filepath):
            print(f"Saved: {filepath}")
            return filepath
        else:
            print(f"Error: File was not created at {filepath}")
            return None
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def record_with_playback(command, speaker_id, command_idx):
    while True:
        audio = record_command(command)
        
        if audio is None or len(audio) == 0:
            print("Recording failed! No audio captured.")
            choice = input("Try again? (y/n): ").strip().lower()
            if choice == 'y':
                continue
            else:
                return

        playback_audio(audio)

        print("\nWhat do you want to do?")
        print("  [y] Save this recording")
        print("  [r] Redo the recording")
        print("  [p] Play it again")

        while True:
            choice = input("Your choice (y/r/p): ").strip().lower()
            if choice == 'p':
                playback_audio(audio)
            elif choice == 'y':
                saved_path = save_command(audio, command, speaker_id, command_idx)
                if saved_path:
                    print("Recording saved successfully!")
                else:
                    print("Failed to save recording. Please try again.")
                    choice = input("Try again? (y/n): ").strip().lower()
                    if choice == 'y':
                        break
                    else:
                        return
                return
            elif choice == 'r':
                print("Redoing...")
                break
            else:
                print("Please enter y, r, or p")

def redo_specific_command(speaker_id):
    print("\nCommand List:")
    for idx, cmd in enumerate(COMMANDS):
        speaker_dir = os.path.join(OUTPUT_DIR, f"speaker_{speaker_id}")
        filename = f"cmd{idx+1:02d}_{cmd.replace(' ', '_')}.wav"
        filepath = os.path.join(speaker_dir, filename)
        status = "OK" if os.path.exists(filepath) else "MISSING"
        print(f"  {status} [{idx+1:02d}] {cmd}")

    while True:
        try:
            num = int(input("\nEnter command number to redo (0 to cancel): "))
            if num == 0:
                break
            if 1 <= num <= len(COMMANDS):
                command = COMMANDS[num - 1]
                print(f"\nRedoing: '{command}'")
                input("Press ENTER when ready...")
                record_with_playback(command, speaker_id, num)
                print("Command updated!")

                another = input("\nRedo another command? (y/n): ").strip().lower()
                if another != 'y':
                    break
            else:
                print(f"Please enter a number between 1 and {len(COMMANDS)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("=" * 50)
    print("   Audio Recording Tool")
    print("=" * 50)

    speaker_id = input("\nEnter speaker name: ").strip().lower()

    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Files will be saved to: {os.path.join(os.getcwd(), OUTPUT_DIR)}")

    print(f"\nHello {speaker_id}! You will record {len(COMMANDS)} commands.")
    print("Each recording is 3 seconds long.")
    print("\nAfter each recording you can:")
    print("  Listen to your recording")
    print("  Save it if it sounds good")
    print("  Redo it if you made a mistake")
    print("  Play it again as many times as you want")
    input("\nPress ENTER when you are ready to start...")

    for idx, command in enumerate(COMMANDS):
        print(f"\n[{idx+1}/{len(COMMANDS)}]")
        record_with_playback(command, speaker_id, idx + 1)

        if idx < len(COMMANDS) - 1:
            input("Press ENTER for next command...")

    print("\nAll commands recorded!")
    redo = input("\nDo you want to redo any specific command? (y/n): ").strip().lower()
    if redo == 'y':
        redo_specific_command(speaker_id)

    print("\nRecording session complete!")
    
    speaker_dir = os.path.join(OUTPUT_DIR, f"speaker_{speaker_id}")
    if os.path.exists(speaker_dir):
        files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        print(f"\nSummary for speaker '{speaker_id}':")
        print(f"   Total files saved: {len(files)}/{len(COMMANDS)}")
        print(f"   Location: {os.path.abspath(speaker_dir)}")
        
        if len(files) < len(COMMANDS):
            missing = len(COMMANDS) - len(files)
            print(f"   Warning: {missing} files are missing!")
    else:
        print(f"Error: Directory {speaker_dir} was not created!")

if __name__ == "__main__":
    main()