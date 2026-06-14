import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLE_RATE = 44100
DURATION = 3

def record_command(filename, label):
    input(f'\nPress Enter to record: {label}...')
    print('Recording... Speak now!')
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print(f'Saved: {filename}')

commands = [
    ('data/raw/speaker_aanchal/cmd01_Check_blood_pressure.wav', 'Check blood pressure'),
    ('data/raw/speaker_aanchal/cmd02_Measure_heart_rate.wav', 'Measure heart rate'),
    ('data/raw/speaker_aanchal/cmd03_Record_temperature.wav', 'Record temperature'),
    ('data/raw/speaker_aanchal/cmd04_Check_oxygen_saturation.wav', 'Check oxygen saturation'),
    ('data/raw/speaker_aanchal/cmd05_Monitor_pulse.wav', 'Monitor pulse'),
    ('data/raw/speaker_aanchal/cmd06_Check_respiratory_rate.wav', 'Check respiratory rate'),
    ('data/raw/speaker_aanchal/cmd07_Start_IV_drip.wav', 'Start IV drip'),
]

for filepath, label in commands:
    record_command(filepath, label)

print('\nAll 7 commands recorded successfully!')