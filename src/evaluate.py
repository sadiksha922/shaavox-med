import json
import os
import torch
import librosa
import numpy as np
import whisper
from jiwer import wer, process_words

# ── Settings ──────────────────────────────────────────
TEST_FILE     = "data/processed/test_split.json"
MODEL_PATH    = "models/shaavox_med.pt"
MODEL_SIZE    = "base"
SAMPLE_RATE   = 16000
RESULTS_FILE  = "evaluation/results.json"
# ──────────────────────────────────────────────────────


def clean_text(t):
    """Lowercase and strip punctuation for fair comparison."""
    return "".join(c for c in t.lower() if c.isalnum() or c.isspace()).strip()


def transcribe_with_model(model, audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = whisper.pad_or_trim(audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


def evaluate():
    print("=" * 50)
    print("   ShaaVox-Med Evaluation")
    print("=" * 50)

    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        print("   Run 'python src/train_model.py' first — it creates the test split.")
        return

    with open(TEST_FILE) as f:
        test_entries = json.load(f)
    print(f"✅ Loaded {len(test_entries)} test samples")

    # ── Evaluate fine-tuned ShaaVox-Med model ──
    print(f"\n⏳ Loading fine-tuned ShaaVox-Med model...")
    finetuned = whisper.load_model(MODEL_SIZE)
    finetuned.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    finetuned.eval()
    print("✅ ShaaVox-Med model loaded!")

    # ── Load baseline (untouched) Whisper for comparison ──
    print(f"⏳ Loading baseline Whisper model (no fine-tuning)...")
    baseline = whisper.load_model(MODEL_SIZE)
    baseline.eval()
    print("✅ Baseline model loaded!")

    results = []
    refs_ft, hyps_ft = [], []
    refs_base, hyps_base = [], []

    print(f"\n🔍 Evaluating {len(test_entries)} samples on both models...")
    for entry in test_entries:
        reference = clean_text(entry['transcript'])

        # Fine-tuned model
        hyp_ft = clean_text(transcribe_with_model(finetuned, entry['audio_path']))
        # Baseline model
        hyp_base = clean_text(transcribe_with_model(baseline, entry['audio_path']))

        refs_ft.append(reference)
        hyps_ft.append(hyp_ft)
        refs_base.append(reference)
        hyps_base.append(hyp_base)

        results.append({
            "audio_path": entry['audio_path'],
            "reference": reference,
            "shaavox_med_hypothesis": hyp_ft,
            "baseline_hypothesis": hyp_base,
            "noise_level": entry.get('noise_level', 'unknown'),
            "shaavox_correct": hyp_ft == reference,
            "baseline_correct": hyp_base == reference,
        })

    # ── Overall metrics ──
    wer_ft = wer(refs_ft, hyps_ft)
    wer_base = wer(refs_base, hyps_base)
    acc_ft = sum(r['shaavox_correct'] for r in results) / len(results)
    acc_base = sum(r['baseline_correct'] for r in results) / len(results)

    print(f"\n📊 Overall Results:")
    print(f"   {'Metric':<28}{'ShaaVox-Med':<15}{'Baseline':<15}")
    print(f"   {'-'*55}")
    print(f"   {'Word Error Rate (WER)':<28}{wer_ft:.2%}{'':<8}{wer_base:.2%}")
    print(f"   {'Command Accuracy':<28}{acc_ft:.2%}{'':<8}{acc_base:.2%}")

    # ── Breakdown by noise level ──
    print(f"\n📊 Breakdown by Noise Level:")
    noise_levels = sorted(set(r['noise_level'] for r in results))
    breakdown = {}
    for level in noise_levels:
        subset = [r for r in results if r['noise_level'] == level]
        sub_acc_ft = sum(r['shaavox_correct'] for r in subset) / len(subset)
        sub_acc_base = sum(r['baseline_correct'] for r in subset) / len(subset)
        breakdown[level] = {
            "count": len(subset),
            "shaavox_accuracy": sub_acc_ft,
            "baseline_accuracy": sub_acc_base,
        }
        print(f"   {level:<15} (n={len(subset):<3}) ShaaVox-Med: {sub_acc_ft:.2%}  |  Baseline: {sub_acc_base:.2%}")

    # ── Save results ──
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            "overall": {
                "shaavox_med_wer": wer_ft,
                "baseline_wer": wer_base,
                "shaavox_med_accuracy": acc_ft,
                "baseline_accuracy": acc_base,
            },
            "by_noise_level": breakdown,
            "details": results,
        }, f, indent=2)

    print(f"\n✅ Full results saved to {RESULTS_FILE}")
    print("🎉 Evaluation complete!")


if __name__ == "__main__":
    evaluate()