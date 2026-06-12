import torch
import whisper
import librosa

print("PyTorch:", torch.__version__)
print("Whisper ready:", whisper is not None)
print("Librosa:", librosa.__version__)
print("GPU available:", torch.cuda.is_available())