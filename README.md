# Real-Time Desktop Audio Translation Overlay

A lightweight, localized Python utility that captures system/speaker audio loops and overlays synchronized real-time English subtitles directly onto your screen. Perfect for translating live video streams, YouTube clips, or foreign media without microphone interference.

## 🚀 Key Features
- **Zero Microphone Interference:** Utilizes Windows WASAPI Loopback to record the digital stream straight from your sound card—completely immune to outside room noise.
- **Sliding Audio Window:** Refreshes subtitles every 1.0 second using a rolling 4-second context window to balance speed and translation accuracy.
- **Edge AI Processing:** Powered by an optimized, int8-quantized `faster-whisper` transformer model running locally on your CPU.
- **Smart Noise Gate:** Employs Root Mean Square (RMS) volume tracking to detect dead air, automatically clearing the screen and preventing AI hallucination loops during silence.
- **Seamless UI:** A borderless, transparent Tkinter overlay that stays "always on top" of your media player or browser. Left-click and drag to reposition; right-click to exit.

## 🛠️ Tech Stack & Concepts
- **Language:** Python
- **Libraries:** `pyaudiowpatch`, `faster-whisper`, `numpy`, `scipy`, `tkinter`
- **Core Concepts:** Multithreading, Thread-Safe Queues, Digital Signal Processing (DSP), Audio Resampling, Window Sliding Algorithms.