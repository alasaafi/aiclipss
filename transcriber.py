import stable_whisper

class Transcriber:
    """Uses Whisper AI to create a transcript with word-level timestamps."""
    def __init__(self, message_queue):
        self.message_queue = message_queue
        # MODIFIED: Use a faster English-only model for speed
        self.model = stable_whisper.load_model('tiny.en')

    def get_word_timestamps(self, audio_path):
        self.message_queue.put(("log", "     🤖 AI Transcribing audio with Whisper..."))
        try:
            # MODIFIED: Set fp16=True for GPU acceleration
            result = self.model.transcribe(audio_path, fp16=True)
            word_list = []
            for segment in result.segments:
                for word in segment.words:
                    word_list.append({
                        "text": word.word.upper().strip(),
                        "start": word.start,
                        "end": word.end
                    })
            self.message_queue.put(("log", "     ✅ AI Transcription complete."))
            return word_list
        except Exception as e:
            self.message_queue.put(("log", f"❌ Whisper transcription failed: {e}"))
            return None