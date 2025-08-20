# transcriber.py

from pathlib import Path
import os
import torch
from faster_whisper import WhisperModel

class Transcriber:
    """
    Transcribes audio using faster-whisper for local, offline processing.
    """
    def __init__(self, message_queue):
        self.message_queue = message_queue
        
        # Determine the compute device (use GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Define the model size. Options: "tiny", "base", "small", "medium", "large-v3"
        # "base" is a good starting point.
        self.model_size = "base"

        self.message_queue.put((f"log", f"🧠 Whisper Transcriber init (model={self.model_size}, device={self.device})"))

        try:
            # This will download the model on the first run and cache it.
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self.message_queue.put(("log", "     ✅ Whisper model loaded successfully."))
        except Exception as e:
            self.message_queue.put(("log", f"     ❌ Failed to load Whisper model: {e}"))
            self.model = None

    def get_word_timestamps(self, audio_path):
        """Returns list of {text, start, end} or None on failure."""
        if not self.model:
            self.message_queue.put(("log", "❌ Transcription failed: Whisper model not loaded."))
            return None
            
        self.message_queue.put(("log", "     🤖 AI Transcribing audio with Whisper..."))
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            self.message_queue.put(("log", f"❌ Transcription failed: audio file not found: {audio_path}"))
            return None

        try:
            # Transcribe the audio and get word-level timestamps
            segments, _ = self.model.transcribe(
                str(audio_file),
                beam_size=5,
                word_timestamps=True
            )

            words = []
            # Iterate through the segments and words to build the final list
            for segment in segments:
                for word in segment.words:
                    words.append({
                        "text": word.word.upper().strip(),
                        "start": float(word.start),
                        "end": float(word.end)
                    })
            
            if not words:
                self.message_queue.put(("log", "     ⚠️ Whisper returned no word timestamps."))
                return []
            
            self.message_queue.put(("log", "     ✅ AI Transcription complete via Whisper."))
            return words

        except Exception as e:
            self.message_queue.put(("log", f"     ❌ Whisper transcription failed: {e}"))
            return None