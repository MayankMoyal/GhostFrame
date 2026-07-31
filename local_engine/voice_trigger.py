import os
import time
import requests
import wave
import pyaudio

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/generate-voice"
PROPS_DIR = "active_props"
RECORD_SECONDS = 5  # How long to record when you press Enter
AUDIO_FILENAME = "temp_voice.wav"

def record_audio():
    """Records audio from the default microphone."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print(f"\n🎙️ Recording for {RECORD_SECONDS} seconds... Speak now!")
    frames = []
    
    for _ in range(0, int(rate / chunk * RECORD_SECONDS)):
        data = stream.read(chunk)
        frames.append(data)

    print("✅ Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def main():
    os.makedirs(PROPS_DIR, exist_ok=True)
    
    print("=======================================")
    print("   Ghost Frame - Voice Trigger UI      ")
    print("=======================================")
    
    while True:
        input("\nPress ENTER to start recording a voice prompt (or Ctrl+C to quit)...")
        
        # 1. Record audio
        record_audio()
        
        # 2. Send to backend
        print("\n🚀 Sending to Cloud Backend for AI processing...")
        start_time = time.time()
        
        try:
            with open(AUDIO_FILENAME, "rb") as f:
                files = {"audio": (AUDIO_FILENAME, f, "audio/wav")}
                data = {"style": "", "remove_bg": "true"}
                response = requests.post(BACKEND_URL, files=files, data=data)
            
            response.raise_for_status()
            result = response.json()
            
        except Exception as e:
            print(f"❌ API Request failed: {e}")
            continue
            
        # 3. Download the final image
        image_name = result.get("filename_nobg") or result.get("filename")
        if not image_name:
            print("❌ Backend did not return a valid image filename.")
            continue
            
        print(f"✅ Backend finished in {time.time() - start_time:.1f}s!")
        print(f"🎯 AI detected category: {result.get('agent', {}).get('anchor_type')}")
        
        # NOTE: If hosting on cloud, replace 127.0.0.1 with your cloud IP
        image_url = f"http://127.0.0.1:8000/outputs/{image_name}"
        print(f"⬇️ Downloading generated prop from {image_url}...")
        
        try:
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            # Save into active_props so ghost_engine.py auto-equips it
            save_path = os.path.join(PROPS_DIR, image_name)
            
            # Optional: Clear old props so you don't accumulate dozens of swords on screen
            for old_file in os.listdir(PROPS_DIR):
                os.remove(os.path.join(PROPS_DIR, old_file))
                
            with open(save_path, "wb") as f:
                f.write(img_response.content)
                
            print(f"🎉 Prop saved to {save_path}. Ghost Engine will equip it instantly!")
            
        except Exception as e:
            print(f"❌ Failed to download image: {e}")

if __name__ == "__main__":
    main()
