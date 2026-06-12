import os
import numpy as np
import librosa
import soundfile as sf

# ── Settings ──────────────────────────────────────────
SAMPLE_RATE = 16000
CLEAN_DIR = "data/raw"
NOISY_DIR = "data/noisy"
NOISE_DIR = "data/noises"
SNR_LEVELS = [20, 10, 0]
# ──────────────────────────────────────────────────────

def add_noise_at_snr(clean_audio, noise_audio, snr_db):
    if len(noise_audio) < len(clean_audio):
        noise_audio = np.tile(noise_audio,
                     int(np.ceil(len(clean_audio)/len(noise_audio))))
    noise_audio = noise_audio[:len(clean_audio)]

    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))

    noisy_audio = clean_audio + scale * noise_audio
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    return noisy_audio.astype(np.float32)

def augment_dataset():
    print("=" * 50)
    print("  ShaaVox-Med Noise Augmentation Tool")
    print("=" * 50)

    noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
    if not noise_files:
        print("❌ No noise files found in data/noises/")
        return

    print(f"✅ Found {len(noise_files)} noise files")
    noise_audios = []
    for nf in noise_files:
        audio, _ = librosa.load(os.path.join(NOISE_DIR, nf), sr=SAMPLE_RATE)
        noise_audios.append(audio)
        print(f"   Loaded: {nf}")

    max_len = max(len(n) for n in noise_audios)
    combined_noise = np.zeros(max_len)
    for n in noise_audios:
        padded = np.tile(n, int(np.ceil(max_len/len(n))))[:max_len]
        combined_noise += padded
    combined_noise = combined_noise / np.max(np.abs(combined_noise))

    total_created = 0
    for speaker in os.listdir(CLEAN_DIR):
        speaker_path = os.path.join(CLEAN_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue

        print(f"\n🎙️  Processing: {speaker}")

        for wav_file in os.listdir(speaker_path):
            if not wav_file.endswith('.wav'):
                continue

            clean_path = os.path.join(speaker_path, wav_file)
            clean_audio, _ = librosa.load(clean_path, sr=SAMPLE_RATE)

            for snr in SNR_LEVELS:
                noisy_audio = add_noise_at_snr(clean_audio, combined_noise, snr)

                out_dir = os.path.join(NOISY_DIR, speaker, f"snr_{snr}db")
                os.makedirs(out_dir, exist_ok=True)

                out_filename = wav_file.replace('.wav', f'_snr{snr}db.wav')
                out_path = os.path.join(out_dir, out_filename)
                sf.write(out_path, noisy_audio, SAMPLE_RATE)
                total_created += 1

        print(f"   ✅ Done!")

    print(f"\n🎉 Augmentation complete!")
    print(f"📁 Total noisy files created: {total_created}")

if __name__ == "__main__":
    augment_dataset()