import whisper
import sys

print("Loading Whisper model...")
model = whisper.load_model("tiny")
print("Model loaded!")

# If you have an audio file, test it
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
    print(f"Transcribing: {audio_file}")
    result = model.transcribe(audio_file)
    print(f"\nResult: {result['text']}")
else:
    print("\nTo test an audio file: python test_transcribe.py your_audio.mp3")