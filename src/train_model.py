import json
import os
import torch
import librosa
import numpy as np
import whisper
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────
DATASET_FILE = "data/processed/dataset.json"
MODEL_SIZE   = "base"
SAMPLE_RATE  = 16000
EPOCHS       = 5
LR           = 1e-5
BATCH_SIZE   = 2
SAVE_PATH    = "models/shaavox_med.pt"
# ──────────────────────────────────────────────────────


class MedicalSpeechDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio, _ = librosa.load(entry['audio_path'], sr=SAMPLE_RATE)
        audio = whisper.pad_or_trim(audio.astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio)
        return mel, entry['transcript']


def collate_fn(batch):
    mels = torch.stack([item[0] for item in batch])
    transcripts = [item[1] for item in batch]
    return mels, transcripts


def train():
    print("=" * 50)
    print("   ShaaVox-Med Model Training")
    print("=" * 50)

    # Load dataset
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        print("   Run 'python src/prepare_dataset.py' first.")
        return

    with open(DATASET_FILE) as f:
        entries = json.load(f)
    print(f"✅ Loaded {len(entries)} samples")

    if len(entries) == 0:
        print("❌ Dataset is empty. Check your recordings and manifest.")
        return

    # Shuffle before splitting so train/test mix speakers and noise levels
    import random
    random.seed(42)
    random.shuffle(entries)

    # Train/test split 80/20
    split = int(0.8 * len(entries))
    train_data = entries[:split]
    test_data = entries[split:]
    print(f"   Train: {len(train_data)} | Test: {len(test_data)}")

    # Save test split for evaluation later
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/test_split.json", "w") as f:
        json.dump(test_data, f, indent=2)
    print("   ✅ Test split saved to data/processed/test_split.json")

    # Load Whisper
    print(f"\n⏳ Loading Whisper '{MODEL_SIZE}' model...")
    model = whisper.load_model(MODEL_SIZE)
    print("✅ Whisper loaded!")

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=False, task="transcribe"
    )

    # Dataset and loader
    dataset = MedicalSpeechDataset(train_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Only fine-tune the decoder (faster, less memory, good for small datasets)
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = Adam(model.decoder.parameters(), lr=LR)

    print(f"\n🚀 Training for {EPOCHS} epochs (decoder fine-tuning only)...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for mels, transcripts in progress:
            optimizer.zero_grad()

            # Encode audio (frozen encoder)
            with torch.no_grad():
                encoded = model.encoder(mels)

            # Build decoder input/target tokens
            sot_sequence = list(tokenizer.sot_sequence)
            token_lists = [
                sot_sequence + tokenizer.encode(t) + [tokenizer.eot]
                for t in transcripts
            ]
            max_len = max(len(t) for t in token_lists)

            input_tokens = torch.zeros(len(token_lists), max_len - 1, dtype=torch.long)
            target_tokens = torch.full((len(token_lists), max_len - 1), -100, dtype=torch.long)

            for i, tokens in enumerate(token_lists):
                seq = torch.tensor(tokens)
                input_tokens[i, :len(seq) - 1] = seq[:-1]
                target_tokens[i, :len(seq) - 1] = seq[1:]

            logits = model.decoder(input_tokens, encoded)

            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), target_tokens, ignore_index=-100
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"   Epoch {epoch+1} avg loss: {avg:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")
    print("🎉 Training complete!")


if __name__ == "__main__":
    train()