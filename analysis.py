# analysis.py
import torch
from transformers import pipeline
from pyannote.audio import Pipeline as PyannotePipeline
import numpy as np
import subprocess
import random 
import re 
import os
import json
import openai # The openai library is used to interact with OpenRouter's compatible API

class ContentAnalyzer:
    def __init__(self, message_queue):
        self.message_queue = message_queue
        self.emotion_pipeline = None
        self.diarization_pipeline = None
        # MODIFIED: Renamed client to be more generic
        self.api_client = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.message_queue.put(("log", f"🧠 AI Analyzer running on: {self.device.upper()}"))
        
        # MODIFIED: Initialize the client for OpenRouter using config.json
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            api_key = config.get("API_KEY")
            
            if not api_key or "your-new-secret" in api_key:
                raise ValueError("API key not found or is still the default placeholder in config.json.")
            
            self.api_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1" # MODIFIED: Using OpenRouter's URL
            )
            self.message_queue.put(("log", "     ✅ API client for OpenRouter initialized successfully."))
        except FileNotFoundError:
            self.api_client = None
            self.message_queue.put(("log", "❌ CRITICAL ERROR: `config.json` file not found. Please create it."))
        except Exception as e:
            self.api_client = None
            self.message_queue.put(("log", f"❌ API client failed to initialize: {e}"))


    def _initialize_pipelines(self):
        """Initializes the Hugging Face models on first use."""
        if self.emotion_pipeline is None:
            self.message_queue.put(("log", "     🧠 Initializing Emotion Analysis pipeline..."))
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=0 if self.device == "cuda" else -1
            )
            self.message_queue.put(("log", "     ✅ Emotion pipeline ready."))

        if self.diarization_pipeline is None:
            self.message_queue.put(("log", "     🧠 Initializing Speaker Diarization pipeline..."))
            try:
                huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
                if not huggingface_token:
                    raise ValueError("Hugging Face token not found in environment variables.")

                self.diarization_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-pipeline",
                    use_auth_token=huggingface_token
                )
                self.diarization_pipeline.to(torch.device(self.device))
                self.message_queue.put(("log", "     ✅ Speaker pipeline ready."))
            except Exception as e:
                self.message_queue.put(("log", f"     ❌ Speaker pipeline failed. Check Hugging Face token. Error: {e}"))
                self.diarization_pipeline = "failed" 
        
    def _analyze_speakers(self, audio_path):
        if self.diarization_pipeline == "failed": return []
        self.message_queue.put(("log", "     🗣️ Analyzing speakers (diarization)..."))
        try:
            diarization = self.diarization_pipeline(audio_path, num_speakers=None)
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            self.message_queue.put((f"log", f"     ✅ Identified {len(speaker_segments)} speaker turns."))
            return speaker_segments
        except Exception as e:
            self.message_queue.put(("log", f"     ❌ Speaker diarization failed: {e}"))
            return []

    def _score_transcript_for_emotion(self, timed_words):
        self.message_queue.put(("log", "     🎭 Scoring transcript for emotional content..."))
        
        full_text = " ".join([word['text'] for word in timed_words])
        sentences = re.split(r'[.?!]\s*', full_text)
        
        sentence_data = []
        word_idx = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence_words = sentence.split()
            max_chunk_words = 200
            
            for i in range(0, len(sentence_words), max_chunk_words):
                chunk_words = sentence_words[i:i + max_chunk_words]
                chunk_text = " ".join(chunk_words)
                num_words_in_chunk = len(chunk_words)
                
                start_time = timed_words[word_idx]['start'] if word_idx < len(timed_words) else timed_words[-1]['end']
                end_word_idx = min(word_idx + num_words_in_chunk - 1, len(timed_words) - 1)
                end_time = timed_words[end_word_idx]['end']
                
                sentence_data.append({"text": chunk_text, "start": start_time, "end": end_time})
                word_idx += num_words_in_chunk

        if not sentence_data:
            return []

        target_emotions = {'joy', 'surprise', 'anger', 'sadness'}
        texts = [s['text'] for s in sentence_data]
        
        batch_size = 8
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_results = self.emotion_pipeline(batch_texts)
                results.extend(batch_results)
            except Exception as e:
                self.message_queue.put(("log", f"     ❌ Error processing text batch: {e}"))
                results.extend([[] for _ in batch_texts])
                
        for i, (scores, data) in enumerate(zip(results, sentence_data)):
            if not scores:
                emotional_score = 0
            else:
                emotional_score = sum(emotion['score'] for emotion in scores if emotion['label'] in target_emotions)
            sentence_data[i]['score'] = emotional_score
        
        self.message_queue.put(("log", "     ✅ Emotional scoring complete."))
        return sentence_data

    def generate_title_and_description(self, clip_words):
        if not self.api_client or not clip_words:
            self.message_queue.put(("log", "     ⚠️ Skipping title generation: API client not available."))
            return {"title": "AI Clip", "description": "A fascinating moment from the video. #aiclip #short"}

        transcript = " ".join([word['text'] for word in clip_words])
        if len(transcript) > 4000:
            transcript = transcript[:4000]

        prompt = f"""
        Based on the following video transcript, please generate a YouTube video title and description.

        **Rules:**
        1. The title must be catchy, engaging, and under 60 characters.
        2. The description should be a brief, one-sentence summary of the clip's content.
        3. The description must end with exactly 3 relevant, popular hashtags.
        4. Your response MUST be a valid JSON object with two keys: "title" and "description".

        **Transcript:**
        "{transcript}"
        """

        try:
            self.message_queue.put(("log", "     ✍️ Contacting OpenRouter to generate title & description..."))
            
            # MODIFIED: Using OpenRouter. You can change the model to any one you like from their website.
            # For example: "mistralai/mistral-7b-instruct-v0.2" or "google/gemma-7b-it"
            # Using a free model as a default.
            response = self.api_client.chat.completions.create(
                model="mistralai/mistral-7b-instruct", 
                messages=[
                    {"role": "system", "content": "You are an expert social media manager who creates viral video titles and descriptions. You always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            metadata = json.loads(content)
            
            if 'title' not in metadata or 'description' not in metadata:
                 raise ValueError("API response did not contain 'title' or 'description' keys.")

            return metadata

        except Exception as e:
            self.message_queue.put(("log", f"     ❌ OpenRouter title/description generation failed: {e}"))
            return {"title": "AI Generated Clip", "description": f"An interesting clip. #video #clip #AI"}

    def find_best_clips(self, audio_path, timed_words, num_clips, clip_duration_arg):
        self._initialize_pipelines()
        
        if not timed_words:
            self.message_queue.put(("log", "     ⚠️ No transcript available. Cannot generate clips."))
            return []
        
        if isinstance(clip_duration_arg, tuple):
            min_dur, max_dur = clip_duration_arg
        else:
            min_dur, max_dur = clip_duration_arg, clip_duration_arg

        video_duration = timed_words[-1]['end']
        speaker_segments = self._analyze_speakers(audio_path)
        scored_sentences = self._score_transcript_for_emotion(timed_words)

        potential_clips = []
        window_step = 5
        for start_time in np.arange(0, video_duration - max_dur, window_step):
            current_clip_duration = random.randint(min_dur, max_dur)
            end_time = start_time + current_clip_duration
            
            if end_time > video_duration: continue

            current_score = 0
            for sentence in scored_sentences:
                overlap_start = max(start_time, sentence['start'])
                overlap_end = min(end_time, sentence['end'])
                if overlap_end > overlap_start:
                    duration = overlap_end - overlap_start
                    current_score += sentence['score']

            speaker_durations = {}
            for segment in speaker_segments:
                overlap_start = max(start_time, segment['start'])
                overlap_end = min(end_time, segment['end'])
                if overlap_end > overlap_start:
                    duration = overlap_end - overlap_start
                    speaker_durations[segment['speaker']] = speaker_durations.get(segment['speaker'], 0) + duration

            dominant_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else None
            if dominant_speaker: current_score *= 1.2

            potential_clips.append({
                'start': start_time, 
                'end': end_time, 
                'score': current_score,
                'dominant_speaker': dominant_speaker
            })

        if not potential_clips:
            self.message_queue.put(("log", "     ❌ No potential clips could be scored."))
            return []

        sorted_clips = sorted(potential_clips, key=lambda x: x['score'], reverse=True)
        final_clips = []
        for clip in sorted_clips:
            if len(final_clips) >= num_clips: break
            is_overlapping = any(max(final_clip['start'], clip['start']) < min(final_clip['end'], clip['end']) for final_clip in final_clips)
            if not is_overlapping:
                final_clips.append(clip)

        self.message_queue.put((f"log", f"     ✅ Identified {len(final_clips)} high-impact clips!"))
        return sorted(final_clips, key=lambda x: x['start'])