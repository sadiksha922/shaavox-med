import streamlit as st
import torch
import whisper
import numpy as np
import sounddevice as sd
import noisereduce as nr
import time
import json
import os

# ── Settings ──────────────────────────────────────────
MODEL_SIZE  = "base"
MODEL_PATH  = "models/shaavox_med.pt"
SAMPLE_RATE = 16000
DURATION    = 3
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

def get_speaker_count():
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        return 0
    return len([f for f in os.listdir(raw_dir) 
                if os.path.isdir(os.path.join(raw_dir, f)) 
                and f.startswith("speaker_")])

def get_dataset_count():
    manifest = "data/processed/dataset.json"
    if not os.path.exists(manifest):
        return 0
    with open(manifest) as f:
        return len(json.load(f))



CATEGORY_MAP = {
    "🫀 Vital Signs":       ["check blood pressure", "measure heart rate", "record temperature", "check oxygen saturation", "monitor pulse", "check respiratory rate"],
    "⚙️ Equipment Control": ["start iv drip", "stop iv drip", "increase oxygen flow", "decrease oxygen flow", "start ventilator", "stop ventilator", "silence alarm", "reset monitor"],
    "🧑‍⚕️ Patient Status":  ["patient is stable", "patient is critical", "patient needs medication", "call the nurse", "request doctor"],
    "💊 Drug Names":        ["paracetamol", "amoxicillin", "ibuprofen", "morphine", "insulin", "metformin"],
    "🏥 Procedural":        ["begin procedure", "end procedure", "prepare operating room", "request blood sample", "start ecg"],
}

@st.cache_resource
def load_model():
    model = whisper.load_model(MODEL_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
    model.eval()
    return model

def match_command(transcript):
    import difflib
    t = transcript.lower().strip()
    
    # Remove common filler words Whisper adds
    fillers = ["thank you", "thanks", "okay", "ok", "um", "uh", 
               "please", "the", "a", "and", "i", "you", "me"]
    cleaned = " ".join(w for w in t.split() if w not in fillers)
    if not cleaned:
        cleaned = t

    # 1. Exact match on cleaned
    if cleaned in COMMANDS:
        return cleaned, "Exact"

    # 2. Exact match on original
    if t in COMMANDS:
        return t, "Exact"

    # 3. Best similarity score across ALL commands (not first match)
    best_cmd, best_score = None, 0
    for cmd in COMMANDS:
        # Full phrase similarity
        ratio = difflib.SequenceMatcher(None, cleaned, cmd).ratio()
        # Word overlap bonus
        overlap = len(set(cleaned.split()) & set(cmd.split()))
        combined = ratio + (overlap * 0.3)
        if combined > best_score:
            best_score = combined
            best_cmd = cmd

    if best_score > 0.35:
        match_type = "Exact" if best_score > 0.85 else "Partial" if best_score > 0.6 else "Fuzzy"
        return best_cmd, match_type

    return None, "None"


def get_category(command):
    for cat, cmds in CATEGORY_MAP.items():
        if command in cmds:
            return cat
    return "Unknown"

def record_and_transcribe(model):
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    audio = audio.flatten()

    # Stronger noise reduction
    noise_sample = audio[:SAMPLE_RATE // 2]
    audio = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        y_noise=noise_sample,
        prop_decrease=0.9,      # stronger reduction
        stationary=True,        # treats noise as constant background
        n_std_thresh_stationary=1.2
    )

    # Normalize audio volume
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Transcribe
    audio = whisper.pad_or_trim(audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(
        language="en",
        fp16=False,
        temperature=0.0,        # greedy decoding — more consistent
        without_timestamps=True
    )
    result = whisper.decode(model, mel, options)
    return result.text.strip()

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="ShaaVox-Med",
    page_icon="🏥",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────
st.title("🏥 ShaaVox-Med")
st.markdown("**Domain-Specific, Noise-Robust Real-Time Medical Speech Recognition**")
st.markdown("---")

# ── Load Model ─────────────────────────────────────────
with st.spinner("⏳ Loading ShaaVox-Med model..."):
    model = load_model()
st.success("✅ Model loaded and ready!")
st.markdown("---")

# ── Metrics ────────────────────────────────────────────
st.subheader("📊 System Overview")
c1, c2, c3, c4 = st.columns(4)
speaker_count = get_speaker_count()
dataset_count = get_dataset_count()
c1.metric("Medical Commands", "30")
c2.metric("Speakers Trained", str(speaker_count))
c3.metric("Word Error Rate", "56.76%", delta="-14.41% vs baseline", delta_color="inverse")
c4.metric("Command Accuracy", "37.50%", delta="+10.42% vs baseline")
st.markdown("---")

# ── Main Layout ────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("🎙️ Voice Recognition")
    st.write("Click **Record** and speak one of the 30 medical commands clearly.")

    if "history" not in st.session_state:
        st.session_state.history = []

    record_btn = st.button(
        "🔴  Record Command  (3 seconds)",
        use_container_width=True,
        type="primary"
    )

    if record_btn:
        with st.spinner("🔴 Recording — speak now!"):
            transcript = record_and_transcribe(model)

        matched_cmd, match_type = match_command(transcript)
        category = get_category(matched_cmd) if matched_cmd else "Unknown"

        st.markdown("#### Result")

        col_a, col_b = st.columns(2)
        col_a.info(f"📝 **Heard:**\n\n_{transcript}_")

        if matched_cmd:
            icon = "✅" if match_type == "Exact" else "🔶"
            col_b.success(
                f"{icon} **{match_type} Match**\n\n"
                f"**{matched_cmd.upper()}**\n\n"
                f"_{category}_"
            )
        else:
            col_b.error("❌ No matching command found.\nPlease speak more clearly.")

        st.session_state.history.insert(0, {
            "time": time.strftime("%H:%M:%S"),
            "transcript": transcript,
            "command": matched_cmd or "No match",
            "match_type": match_type,
            "category": category,
        })

    # History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📋 Session History")
        for item in st.session_state.history[:10]:
            st.markdown(
                f"🕐 `{item['time']}` &nbsp;|&nbsp; "
                f"**{item['command'].upper()}** &nbsp;|&nbsp; "
                f"_{item['transcript']}_ &nbsp;|&nbsp; "
                f"`{item['match_type']}`"
            )
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

with right:
    st.subheader("📋 Command Vocabulary")
    for cat, cmds in CATEGORY_MAP.items():
        with st.expander(f"{cat}  ({len(cmds)} commands)"):
            for cmd in cmds:
                st.markdown(f"• {cmd}")

    st.markdown("---")
    st.subheader("📈 Evaluation Results")

    if os.path.exists("evaluation/results.json"):
        with open("evaluation/results.json") as f:
            eval_data = json.load(f)
        overall = eval_data["overall"]

        st.metric(
            "ShaaVox-Med WER",
            f"{overall['shaavox_med_wer']:.2%}",
            delta=f"{overall['baseline_wer'] - overall['shaavox_med_wer']:.2%} improvement vs baseline",
            delta_color="inverse"
        )
        st.metric(
            "Command Accuracy",
            f"{overall['shaavox_med_accuracy']:.2%}",
            delta=f"+{overall['shaavox_med_accuracy'] - overall['baseline_accuracy']:.2%} vs baseline"
        )

        st.markdown("**Accuracy by Noise Level**")
        for level, data in eval_data["by_noise_level"].items():
            pct = data["shaavox_accuracy"]
            st.progress(pct, text=f"{level}: {pct:.1%}")
    else:
        st.info("Run evaluate.py to see results here.")

    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.markdown(f"""
| Setting | Value |
|---|---|
| Model | Whisper {MODEL_SIZE} (fine-tuned) |
| Vocabulary | 30 commands |
| SNR Levels | 0 / 10 / 20 dB |
| Sample Rate | {SAMPLE_RATE} Hz |
| Duration | {DURATION} seconds |
| Noise Reduction | ✅ Enabled |
    """)