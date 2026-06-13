from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
import threading
from datetime import datetime
import sys

# Print Python path for debugging
print(f"Python path: {sys.executable}")

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper imported successfully")
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"❌ Whisper import failed: {e}")

app = Flask(__name__)
CORS(app)

DATA_DIR = "dictations"
os.makedirs(DATA_DIR, exist_ok=True)

dictations = {}
segments = {}

# Load Whisper
whisper_model = None
if WHISPER_AVAILABLE:
    print("Loading Whisper model...")
    try:
        whisper_model = whisper.load_model("tiny")  # Using tiny for faster loading
        print("✅ Whisper model loaded!")
    except Exception as e:
        print(f"❌ Whisper load error: {e}")
        whisper_model = None

def load_data():
    global dictations, segments
    dictations_file = os.path.join(DATA_DIR, "dictations.json")
    segments_file = os.path.join(DATA_DIR, "segments.json")
    
    if os.path.exists(dictations_file):
        with open(dictations_file, 'r') as f:
            dictations = json.load(f)
    if os.path.exists(segments_file):
        with open(segments_file, 'r') as f:
            segments = json.load(f)

def save_data():
    with open(os.path.join(DATA_DIR, "dictations.json"), 'w') as f:
        json.dump(dictations, f)
    with open(os.path.join(DATA_DIR, "segments.json"), 'w') as f:
        json.dump(segments, f)

load_data()

def transcribe_audio(dict_id, audio_path):
    """Transcribe with detailed logging"""
    print(f"\n{'='*50}")
    print(f"🎤 Starting transcription for: {dict_id}")
    print(f"📁 Audio path: {audio_path}")
    print(f"📁 File exists: {os.path.exists(audio_path)}")
    print(f"📁 File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
    
    try:
        if whisper_model is None:
            print("⚠️ No Whisper model, using simulation")
            # Fallback to simulation
            time.sleep(2)
            sample_segments = [
                {"id": str(uuid.uuid4())[:8], "start_time": 0, "text": "This is a simulated transcription because Whisper is not available.", "is_corrected": False},
                {"id": str(uuid.uuid4())[:8], "start_time": 5, "text": "Please install OpenAI Whisper for real transcription.", "is_corrected": False}
            ]
            segments[dict_id] = sample_segments
            dictations[dict_id]['status'] = 'completed'
            dictations[dict_id]['duration_seconds'] = 10
            save_data()
            print("✅ Simulation complete")
            return
        
        # Real transcription
        print("🎤 Transcribing with Whisper...")
        result = whisper_model.transcribe(audio_path)
        print(f"📝 Transcription result: {result['text'][:100]}...")
        
        # Create segments
        transcript_segments = []
        for i, seg in enumerate(result['segments']):
            transcript_segments.append({
                'id': str(uuid.uuid4())[:8],
                'start_time': seg['start'],
                'text': seg['text'].strip(),
                'is_corrected': False
            })
        
        segments[dict_id] = transcript_segments
        dictations[dict_id]['status'] = 'completed'
        dictations[dict_id]['duration_seconds'] = int(result['segments'][-1]['end']) if result['segments'] else 0
        save_data()
        
        print(f"✅ Transcription complete! {len(transcript_segments)} segments")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        dictations[dict_id]['status'] = 'error'
        save_data()

@app.route('/dictations', methods=['GET'])
def get_dictations():
    return jsonify(list(dictations.values()))

@app.route('/dictations', methods=['POST'])
def create_dictation():
    data = request.json
    dict_id = str(uuid.uuid4())[:8]
    
    dictation = {
        'id': dict_id,
        'title': data.get('title', 'Untitled Dictation'),
        'surgeon_name': data.get('surgeon_name'),
        'procedure_type': data.get('procedure_type'),
        'patient_ref': data.get('patient_ref'),
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'duration_seconds': None,
        'audio_url': None
    }
    
    dictations[dict_id] = dictation
    save_data()
    print(f"📝 Created dictation: {dict_id} - {dictation['title']}")
    return jsonify({'id': dict_id})

@app.route('/dictations/<dict_id>/audio', methods=['POST'])
def upload_audio(dict_id):
    print(f"\n📥 Received audio upload for: {dict_id}")
    
    if dict_id not in dictations:
        return jsonify({'error': 'Dictation not found'}), 404
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save audio file
    filename = f"{dict_id}_{file.filename}"
    filepath = os.path.join(DATA_DIR, filename)
    file.save(filepath)
    print(f"💾 Saved audio to: {filepath}")
    print(f"📏 File size: {os.path.getsize(filepath)} bytes")
    
    # Update dictation
    dictations[dict_id]['audio_url'] = f"/audio/{filename}"
    dictations[dict_id]['status'] = 'transcribing'
    save_data()
    
    # Start transcription in background
    thread = threading.Thread(target=transcribe_audio, args=(dict_id, filepath))
    thread.daemon = True
    thread.start()
    print(f"🔄 Started transcription thread for {dict_id}")
    
    return jsonify({'id': dict_id, 'status': 'transcribing'})

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(DATA_DIR, filename))

@app.route('/dictations/<dict_id>', methods=['GET'])
def get_dictation(dict_id):
    if dict_id not in dictations:
        return jsonify({'error': 'Not found'}), 404
    
    dictation = dictations[dict_id].copy()
    dictation['segments'] = segments.get(dict_id, [])
    print(f"📤 Sending dictation {dict_id}: status={dictation['status']}, segments={len(dictation['segments'])}")
    return jsonify(dictation)

@app.route('/segments/<segment_id>', methods=['PUT'])
def update_segment(segment_id):
    for dict_id, segs in segments.items():
        for seg in segs:
            if seg['id'] == segment_id:
                data = request.json
                seg['text'] = data['text']
                seg['is_corrected'] = True
                save_data()
                return jsonify({'success': True})
    return jsonify({'error': 'Segment not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'whisper_available': WHISPER_AVAILABLE and whisper_model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SURGICAL DICTATION BACKEND (DEBUG MODE)")
    print("="*50)
    print(f"📍 Server: http://localhost:8000")
    print(f"📁 Data dir: {os.path.abspath(DATA_DIR)}")
    print(f"🎤 Whisper: {'✅ Available' if whisper_model else '❌ Not available'}")
    print("="*50 + "\n")
    app.run(debug=True, port=8000, use_reloader=False)