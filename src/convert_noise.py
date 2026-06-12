import librosa
import soundfile as sf
import os

NOISE_DIR = "data/noises"
SAMPLE_RATE = 16000

for f in os.listdir(NOISE_DIR):
    if f.endswith(".mp3"):
        mp3_path = os.path.join(NOISE_DIR, f)
        wav_path = os.path.join(NOISE_DIR, f.replace(".mp3", ".wav"))

        audio, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"✅ Converted: {f} → {os.path.basename(wav_path)}")