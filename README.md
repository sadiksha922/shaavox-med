This is our Final year project.
# shaavox-med

## How to use this script
**Step 1:** Install packages
**Step 2:** Run script
**Step 3:** Enter speaker name when asked
**Step 4:** Wait for countdown (3...2...1...)
**Step 5:** Speak the command clearly when it says "RECORDING NOW!"
**Step 6:** Listen to playback
**Step 7:** Choose option:
- y = save
- r = redo
- p = play again
**Step 8:** Press ENTER for next command
**Step 9:** Repeat steps 4-8 for all 30 commands
**Step 10:** Type y to redo any command if needed, or n to finish
**Step 11:** Find recordings in `data/raw/speaker_YOURNAME/`


**Install packages:**
- pip install sounddevice
- pip install scipy
- pip install numpy
- pip install librosa
- pip install soundfile

Or install all at once: `pip install sounddevice scipy numpy librosa soundfile`

**Steps:**
(1) Run `python record_audio.py` (2) Enter your speaker name (3) Wait for 3 second countdown (4) Speak the command when it says "RECORDING NOW!" (5) Listen to playback (6) Press y to save, r to redo, or p to play again (7) Press ENTER for next command (8) Repeat for all 30 commands (9) Type y to redo any command at the end or n to finish (10) Find recordings in `data/raw/speaker_YOURNAME/`
