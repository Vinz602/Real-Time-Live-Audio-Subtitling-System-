import os
# Fix the Intel OpenMP duplicate engine error immediately
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyaudiowpatch as pyaudio
import tkinter as tk
import threading
import queue
import scipy.signal as signal
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
MODEL_SIZE = "tiny"        # Optimized for speed on laptop CPUs
SAMPLE_RATE = 16000        # Target Whisper sample rate
STEP_DURATION = 1.0        # How often we update the screen (1 second)
CONTEXT_DURATION = 4.0     # Total audio context given to Whisper for accuracy
SILENCE_THRESHOLD = 0.005  # Volume threshold. Anything below this is ignored as dead air.

text_queue = queue.Queue()

print("Loading Optimized Whisper Model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully!")

def audio_translation_worker():
    """Captures audio and uses a rolling context buffer with a noise gate filter."""
    p = pyaudio.PyAudio()
    
    try:
        default_speakers = p.get_default_output_device_info()
    except IOError:
        print("Error: No default speakers found.")
        p.terminate()
        return

    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    loopback_device = None
    
    for device_index in range(0, p.get_device_count()):
        dev_info = p.get_device_info_by_index(device_index)
        if dev_info["hostApi"] == wasapi_info["index"] and dev_info["maxInputChannels"] > 0:
            if default_speakers["name"] in dev_info["name"]:
                loopback_device = dev_info
                break

    if loopback_device is None:
        print("Could not locate WASAPI Loopback device.")
        p.terminate()
        return

    hardware_sample_rate = int(loopback_device["defaultSampleRate"])
    hardware_channels = loopback_device["maxInputChannels"]
    
    # Calculate sizes in bytes (float32 = 4 bytes per sample)
    bytes_per_second = hardware_sample_rate * hardware_channels * 4
    step_bytes_needed = int(bytes_per_second * STEP_DURATION)
    max_context_bytes = int(bytes_per_second * CONTEXT_DURATION)

    print(f"\n--- Performance Engine Active ---")
    print(f"Refreshing subtitles every: {STEP_DURATION} second(s)")
    print(f"Rolling Audio Context: {CONTEXT_DURATION} seconds")
    print(f"Silence Gate Threshold: {SILENCE_THRESHOLD}")
    print(f"----------------------------------\n")

    audio_stream_buffer = b""
    buffer_lock = threading.Lock()

    def callback(in_data, frame_count, time_info, status):
        nonlocal audio_stream_buffer
        with buffer_lock:
            audio_stream_buffer += in_data
            if len(audio_stream_buffer) > max_context_bytes * 2:
                audio_stream_buffer = audio_stream_buffer[-max_context_bytes * 2:]
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=hardware_channels,
        rate=hardware_sample_rate,
        input=True,
        input_device_index=loopback_device["index"],
        frames_per_buffer=1024,
        stream_callback=callback
    )

    stream.start_stream()
    rolling_audio_bytes = b""

    while stream.is_active():
        with buffer_lock:
            if len(audio_stream_buffer) >= step_bytes_needed:
                new_data = audio_stream_buffer
                audio_stream_buffer = b""
            else:
                new_data = b""

        if len(new_data) > 0:
            rolling_audio_bytes += new_data
            
            if len(rolling_audio_bytes) > max_context_bytes:
                rolling_audio_bytes = rolling_audio_bytes[-max_context_bytes:]
            
            try:
                audio_np = np.frombuffer(rolling_audio_bytes, dtype=np.float32)
                audio_np = audio_np.reshape(-1, hardware_channels)
                
                if hardware_channels > 1:
                    audio_mono = np.mean(audio_np, axis=1)
                else:
                    audio_mono = audio_np.flatten()
                
                # --- NOISE GATE / SILENCE DETECTION ---
                # Check the average volume level of the current block
                volume_level = np.sqrt(np.mean(audio_mono**2))
                
                if volume_level < SILENCE_THRESHOLD:
                    # Clear out the text on screen if it's completely quiet
                    text_queue.put("") 
                    threading.Event().wait(STEP_DURATION)
                    continue
                # --------------------------------------

                # Resample down to 16kHz for Whisper
                num_samples = int(len(audio_mono) * SAMPLE_RATE / hardware_sample_rate)
                resampled_audio = signal.resample(audio_mono, num_samples).astype(np.float32)
                
                segments, info = model.transcribe(
                    resampled_audio, 
                    beam_size=3,        
                    task="transcribe",
                    temperature=0.0     
                )
                
                text_output = " ".join([segment.text for segment in segments]).strip()
                
                if text_output:
                    print(f" Live Stream Text -> {text_output}")
                    text_queue.put(text_output)
                    
            except Exception as e:
                print(f"Processing Error: {e}")

        threading.Event().wait(STEP_DURATION)

    stream.stop_stream()
    stream.close()
    p.terminate()

def update_gui_text(label, root):
    try:
        while True:
            new_text = text_queue.get_nowait()
            label.config(text=new_text)
    except queue.Empty:
        pass
    root.after(100, lambda: update_gui_text(label, root))

def create_overlay():
    root = tk.Tk()
    root.title("Live Subtitles")
    root.overrideredirect(True)
    root.wm_attributes("-topmost", True)
    root.config(bg='black')
    root.wm_attributes("-transparentcolor", "black")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{int(screen_width * 0.8)}x100+{int(screen_width * 0.1)}+{int(screen_height * 0.8)}")
    
    label = tk.Label(
        root, 
        text="[System Active - Play Video]", 
        font=("Helvetica", 24, "bold"), 
        fg="white", 
        bg="black", 
        wraplength=int(screen_width * 0.75)
    )
    label.pack(expand=True, fill='both')

    def start_move(event): root.x = event.x; root.y = event.y
    def stop_move(event):
        x = root.winfo_x() + event.x - root.x
        y = root.winfo_y() + event.y - root.y
        root.geometry(f"+{x}+{y}")
        
    label.bind("<Button-1>", start_move)
    label.bind("<B1-Motion>", stop_move)
    label.bind("<Button-3>", lambda e: root.destroy())

    threading.Thread(target=audio_translation_worker, daemon=True).start()
    root.after(100, lambda: update_gui_text(label, root))
    root.mainloop()

if __name__ == "__main__":
    create_overlay()