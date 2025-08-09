# main_logic.py

import threading
from pathlib import Path
from yt_dlp import YoutubeDL
import os
import subprocess
import sys
import queue
import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
from PIL import Image
import re
import shutil
import json
import torch
from transformers import pipeline

from caption_styles import CaptionGenerator
from transcriber import Transcriber
from analysis import ContentAnalyzer

# TARGET_RESOLUTION is now dynamic, passed from app.py
DEFAULT_TARGET_RESOLUTION = (1080, 1920)

class YouTubeClipperCore:
    """Core logic for downloading, analyzing, and clipping YouTube videos."""
    def __init__(self, log_callback, target_resolution=DEFAULT_TARGET_RESOLUTION):
        self.log_callback = log_callback
        self.TARGET_RESOLUTION = target_resolution
        self._internal_message_queue = queue.Queue()

        self.caption_generator = CaptionGenerator(self._internal_message_queue)
        self.transcriber = None
        self.analyzer = None
        self.vignette_path = None
        self.final_clip_metadata = []

        threading.Thread(target=self._process_internal_queue, daemon=True).start()

    def _get_transcriber(self):
        """Initializes and returns the transcriber instance."""
        if self.transcriber is None:
            self.log_callback("log", "🧠 Initializing AI Transcription Engine...")
            self.transcriber = Transcriber(self._internal_message_queue)
        return self.transcriber

    def _get_analyzer(self):
        """Initializes and returns the content analyzer instance."""
        if self.analyzer is None:
            self.log_callback("log", "🧠 Initializing AI Content Analyzer...")
            self.analyzer = ContentAnalyzer(self._internal_message_queue)
        return self.analyzer
    
    def _process_internal_queue(self):
        """Processes messages from the internal queue and sends them to the main app log."""
        while True:
            try:
                item = self._internal_message_queue.get()
                msg_type, value = (item[0], item[1] if len(item) > 1 else None)
                self.log_callback(msg_type, value)
            except Exception as e:
                self.log_callback("log", f"ERROR processing internal queue: {e}")
            finally:
                self._internal_message_queue.task_done()

    def _intelligent_clipping_pipeline(self, url, num_clips, clip_duration_arg, add_captions, font_choice, caption_style, download_folder_path, temp_folder_path):
        """
        Main pipeline to download, analyze, and generate clips from a YouTube video.
        """
        self.log_callback("log", "🧹 Cleaning up old files (for this session)...")
        self.log_callback("progress", 5)

        download_folder_path.mkdir(exist_ok=True, parents=True)
        temp_folder_path.mkdir(exist_ok=True, parents=True)
        
        for folder in [download_folder_path, temp_folder_path]:
            for f in folder.glob('*'):
                try: f.unlink()
                except OSError as e: self.log_callback("log", f"⚠️ Could not remove {f.name}: {e}")
        
        self.final_clip_metadata = []
        
        full_video_path, full_audio_path = None, None
        try:
            self.log_callback("log", "🚀 Downloading full video for analysis...")
            self.log_callback("progress", 10)
            ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': str(download_folder_path / 'full_source_video.%(ext)s'), 'merge_output_format': 'mp4'}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_filepath = ydl.prepare_filename(info)
            full_video_path = Path(downloaded_filepath) if downloaded_filepath and Path(downloaded_filepath).exists() else None
            if not full_video_path: raise FileNotFoundError("Downloaded video file not found.")
            self.log_callback("log", f"✅ Full video downloaded: {full_video_path.name}")
            self.log_callback("progress", 20)
            
            # Correction : Générer la vignette de la source ici
            self.log_callback("log", "📸 Generating source video thumbnail...")
            self._create_source_thumbnail(full_video_path, download_folder_path)
            self.log_callback("progress", 22)
            
            self.log_callback("log", "🔊 Extracting full audio for transcription...")
            self.log_callback("progress", 25)
            full_audio_path = full_video_path.with_suffix('.mp3')
            subprocess.run(["ffmpeg", "-i", str(full_video_path), "-vn", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-q:a", "2", str(full_audio_path)], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.log_callback("progress", 30)

            timed_words = self._get_transcriber().get_word_timestamps(str(full_audio_path))
            if timed_words is None:
                raise Exception("Transcription failed.")
            self.log_callback("progress", 40)

            best_clips = self._get_analyzer().find_best_clips(str(full_audio_path), timed_words, num_clips, clip_duration_arg)
            if not best_clips:
                self.log_callback("log", "Could not find any suitable clips based on AI analysis.")
                return

            self._create_vignette(self.TARGET_RESOLUTION, temp_folder_path)

            for i, clip_info in enumerate(best_clips):
                speaker_log = f"(Speaker: {clip_info.get('dominant_speaker', 'N/A')})"
                self.log_callback("log", f"\n--- Processing Clip {i+1}/{len(best_clips)} (Score: {clip_info['score']:.2f}) {speaker_log} ---")
                
                clip_words = [word.copy() for word in timed_words if clip_info['start'] <= word['start'] < clip_info['end']]
                for word in clip_words:
                    word['start'] -= clip_info['start']
                    word['end'] -= clip_info['start']
                
                # Use a very short, generic name for the intermediate file to avoid WinError 206
                temp_output_filename = f"clip_{i+1}_raw_output.mp4"
                
                self._process_single_clip_with_ffmpeg(str(full_video_path), clip_info, temp_output_filename, add_captions, font_choice, caption_style, clip_words, download_folder_path, temp_folder_path)
                
                # New: Create a thumbnail for the clip
                temp_output_path = download_folder_path / temp_output_filename
                self._create_video_thumbnail(temp_output_path, download_folder_path)

                self.log_callback("log", "     ✍️ Generating AI title and description...")
                metadata = self._get_analyzer().generate_title_and_description(clip_words)
                
                # Sanitize and TRUNCATE the title for the filename to avoid WinError 206
                sanitized_title = re.sub(r'[\\/*?:"<>|]', "", metadata['title'])
                
                # TRUNCATE THE SANITIZED TITLE TO A REASONABLE LENGTH (e.g., 30-40 characters)
                max_filename_title_length = 35
                if len(sanitized_title) > max_filename_title_length:
                    sanitized_title = sanitized_title[:max_filename_title_length].strip()
                
                final_filename = f"clip_{i+1}_{sanitized_title}.mp4"
                
                temp_path = download_folder_path / temp_output_filename
                final_path = download_folder_path / final_filename
                if temp_path.exists():
                    try:
                        temp_path.rename(final_path)
                        self.log_callback("log", f"     ✅ AI Title: \"{metadata['title']}\"")
                        self.log_callback("log", f"     ✅ Renamed file to: {final_filename}")
                        
                        metadata['filename'] = final_filename
                        self.final_clip_metadata.append(metadata)
                    except OSError as e:
                        self.log_callback("log", f"     ❌ Failed to rename {temp_output_filename} to {final_filename}: {e}")
                        self.log_callback("log", "     ⚠️ This might be due to a long filename issue.")
                        metadata['filename'] = temp_output_filename
                        self.final_clip_metadata.append(metadata)

                self.log_callback("progress", int(((i + 1) / len(best_clips)) * 100))

            self.log_callback("log", "\n✅ All clips generated successfully!")
            self.log_callback("progress", 100)
        except Exception as e:
            import traceback
            self.log_callback("log", f"❌ A critical error occurred: {e}\n{traceback.format_exc()}")
        finally:
            if full_video_path and full_video_path.exists(): full_video_path.unlink()
            if full_audio_path and full_audio_path.exists(): full_audio_path.unlink()
            if temp_folder_path.exists():
                try:
                    shutil.rmtree(temp_folder_path)
                except OSError as e:
                    self.log_callback("log", f"⚠️ Could not remove temp folder {temp_folder_path}: {e}")
            
            self.log_callback("log", "--- Processing pipeline finished ---")
            
    def _create_source_thumbnail(self, video_path, output_folder):
        """
        Crée une vignette de la vidéo source pour l'affichage en direct.
        Prend une capture d'écran à la première seconde de la vidéo.
        """
        self.log_callback("log", "     📸 Generating source thumbnail...")
        thumbnail_path = output_folder / "source_thumbnail.jpg"
        
        try:
            if thumbnail_path.exists():
                thumbnail_path.unlink()

            command = ["ffmpeg", "-y", "-ss", "00:00:01", "-i", str(video_path), "-vframes", "1", "-q:v", "2", str(thumbnail_path)]
            subprocess.run(command, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.log_callback("log", "     ✅ Source thumbnail generated successfully.")
        except subprocess.CalledProcessError as e:
            self.log_callback("log", f"     ❌ Failed to generate source thumbnail: {e.stderr.decode()}")
        except Exception as e:
            self.log_callback("log", f"     ❌ An error occurred during source thumbnail generation: {e}")

    def _process_single_clip_with_ffmpeg(self, source_video_path, clip_info, output_filename, add_captions, font_choice, caption_style, timed_words, download_folder_path, temp_folder_path):
        start_time = clip_info['start']
        duration = clip_info['end'] - clip_info['start']
        
        self.log_callback("log", "     🔥 Building optimized FFmpeg command...")
        
        video_codec, preset, bitrate = "libx264", "fast", "12000k"
        ffmpeg_cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration), "-i", source_video_path, "-threads", "0"]
        video_filters = []
        
        track_points = self._get_face_track_points(source_video_path, start_time, duration)
        target_aspect = self.TARGET_RESOLUTION[0] / self.TARGET_RESOLUTION[1] 
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0", source_video_path]
        original_h = int(subprocess.check_output(probe_cmd).decode().strip())
        crop_w = int(original_h * target_aspect)
        x_expr = self._build_dynamic_pan_expression([p['x'] for p in track_points], [p['t'] for p in track_points], source_video_path)
        crop_x = f"({x_expr})-{crop_w/2}"
        
        video_filters.append(f"crop=w={crop_w}:h=ih:x='{crop_x}':y=0")
        video_filters.append(f"scale={self.TARGET_RESOLUTION[0]}:{self.TARGET_RESOLUTION[1]}")
        video_filters.append("unsharp=5:5:0.5:5:5:0.0")
        video_filters.append("curves=preset=medium_contrast")
        video_filters.append(f"fade=type=in:duration=0.4")
        video_filters.append(f"fade=type=out:duration=0.4:start_time={duration-0.4}")
        
        filter_complex_str = f"[0:v]{','.join(video_filters)}[base];"
        
        caption_images = []
        if add_captions and timed_words:
            caption_images = self.caption_generator.create_caption_images(timed_words, font_choice, caption_style, temp_folder_path)
        
        overlay_inputs = [self.vignette_path] + [img['path'] for img in caption_images]
        for inp in overlay_inputs:
            if inp: ffmpeg_cmd.extend(["-i", str(inp)])

        last_video_stream = "[base]"
        vignette_input_idx = 1
        if self.vignette_path:
            filter_complex_str += f"{last_video_stream}[{vignette_input_idx}:v]overlay=0:0:format=auto,format=yuv420p[v_with_vignette];"
            last_video_stream = "[v_with_vignette]"

        caption_start_idx = vignette_input_idx + 1 if self.vignette_path else 1
        for i, img_info in enumerate(caption_images):
            stream_idx = i + caption_start_idx
            start_cap, end_cap = img_info['start'], img_info['end']
            
            y_pos = '(H-h)/2'
            overlay_filter = f"overlay=(W-w)/2:{y_pos}:enable='between(t,{start_cap},{end_cap})'"
            
            current_input_stream = last_video_stream
            output_stream_label = f"[v_out_{i}]"
            filter_complex_str += f"{current_input_stream}[{stream_idx}:v]{overlay_filter}{output_stream_label};"
            last_video_stream = output_stream_label
        
        final_map_stream = last_video_stream.replace('[','').replace(']','')
        ffmpeg_cmd.extend(["-filter_complex", filter_complex_str, "-map", f"[{final_map_stream}]", "-map", "0:a"])
        audio_filters = ["loudnorm"]
        ffmpeg_cmd.extend(["-c:v", video_codec, "-preset", preset, "-b:v", bitrate, "-c:a", "aac", "-b:a", "192k", "-af", ",".join(audio_filters)])
        
        output_path = download_folder_path / output_filename
        ffmpeg_cmd.append(str(output_path))
        
        self.log_callback("log", f"     💾 Executing FFmpeg and saving to {output_filename}...")
        self.log_callback("log", "     🔨 This render is optimized and should be faster...")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.log_callback("log", f"     ✅ Successfully generated {output_filename}")
        except subprocess.CalledProcessError as e:
            self.log_callback("log", f"     ❌ FFmpeg failed for {output_filename}:")
            error_log = e.stderr.decode()
            self.log_callback("log", error_log.split('Error parsing global options:')[0])

    def _create_video_thumbnail(self, video_path, output_folder):
        """
        Crée une vignette pour un clip généré à la marque des 3 secondes.
        """
        self.log_callback("log", "     📸 Generating video thumbnail at 3s...")
        thumbnail_path = output_folder / "thumbnail.jpg"
        
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
            duration_str = subprocess.check_output(probe_cmd).decode().strip()
            duration_sec = float(duration_str)
            if duration_sec < 3.0:
                self.log_callback("log", "     ⚠️ Video is less than 3 seconds long. Generating thumbnail from the start.")
                start_time = "00:00:00"
            else:
                start_time = "00:00:03"
        except (subprocess.CalledProcessError, ValueError) as e:
            self.log_callback("log", f"     ❌ Could not determine video duration. Generating thumbnail from the start. Error: {e}")
            start_time = "00:00:00"

        try:
            command = ["ffmpeg", "-y", "-ss", start_time, "-i", str(video_path), "-vframes", "1", "-q:v", "2", str(thumbnail_path)]
            subprocess.run(command, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.log_callback("log", "     ✅ Thumbnail generated successfully.")
        except subprocess.CalledProcessError as e:
            self.log_callback("log", f"     ❌ Failed to generate thumbnail: {e.stderr.decode()}")
        except Exception as e:
            self.log_callback("log", f"     ❌ An error occurred during thumbnail generation: {e}")

    def _get_frame_at_time(self, video_path, time_sec):
        """Extracts a single frame from a video at a specific time."""
        command = ['ffmpeg', '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', '-f', 'image2pipe', '-c:v', 'png', '-']
        try:
            pipe = subprocess.run(command, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            image_data = np.frombuffer(pipe.stdout, np.uint8)
            return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log_callback("log", f"     ❌ Could not get frame for analysis: {e}")
            return None

    def _get_face_track_points(self, video_path, clip_start_time, duration):
        """
        Analyzes video frames to find face positions for dynamic cropping.
        Uses a faster sampling interval for better performance.
        """
        self.log_callback("log", "     🤖 Optimizing face analysis (faster sampling)...")
        points = []
        sample_interval = 1.5
        sample_times = np.unique(np.concatenate(([0], np.arange(1.0, duration, sample_interval), [duration-0.1]))).tolist()
        last_known_x = None
        
        with mp_solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
            for t in sample_times:
                image = self._get_frame_at_time(video_path, clip_start_time + t)
                if image is None: 
                    self.log_callback("log", f"         Skipping face detection for frame at {clip_start_time + t}s (frame not found).")
                    continue
                results = fd.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                face_x_to_add = last_known_x
                if results.detections:
                    face = max(results.detections, key=lambda d: d.score[0])
                    box = face.location_data.relative_bounding_box
                    face_x_to_add = (box.xmin + box.width / 2) * image.shape[1]
                    last_known_x = face_x_to_add
                if face_x_to_add is not None:
                    points.append({'t': t, 'x': face_x_to_add})

        if not points: 
            self.log_callback("log", "     ⚠️ No face tracking points found. Defaulting to center pan.")
            return []
        if points[0]['t'] > 0:
            points.insert(0, {'t': 0, 'x': points[0]['x']})
        self.log_callback("log", f"     ✅ Found {len(points)} face tracking points.")
        return points

    def _build_dynamic_pan_expression(self, values, timestamps, video_path):
        """
        Creates an FFmpeg expression for smooth, dynamic pan/crop based on face tracking.
        """
        if not values:
             probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", video_path]
             w = int(subprocess.check_output(probe_cmd).decode().strip())
             return str(w/2)
        if len(values) < 2:
            return str(values[0])
        expression = ""
        for i in range(len(values) - 1):
            t1, v1 = timestamps[i], values[i]
            t2, v2 = timestamps[i+1], values[i+1]
            if t2 - t1 <= 0: continue
            segment_expr = f"lerp({v1},{v2},(t-{t1})/({t2-t1}))"
            expression += f"{segment_expr}*between(t,{t1},{t2})+"
        return expression.strip('+') + f"+{values[-1]}*gte(t,{timestamps[-1]})"
    
    def _create_vignette(self, resolution, temp_folder_path):
        """
        Creates a dark, transparent vignette image for video overlay.
        """
        self.vignette_path = temp_folder_path / "vignette.png"
        if self.vignette_path.exists():
            self.log_callback("log", "     🖼️ Vignette already exists for this session, skipping creation.")
            return
        self.log_callback("log", "     🖼️ Creating vignette overlay...")
        width, height = resolution
        radial_gradient = np.sqrt(np.linspace(-1, 1, width)[np.newaxis, :]**2 + np.linspace(-1, 1, height)[:, np.newaxis]**2)
        vignette_alpha = 1 - np.clip(radial_gradient * 1.2, 0, 1)
        vignette_image = np.zeros((height, width, 4), dtype=np.uint8)
        vignette_image[:, :, 3] = ((1 - vignette_alpha) * 255 * 0.5).astype(np.uint8)
        
        temp_folder_path.mkdir(exist_ok=True, parents=True)
        Image.fromarray(vignette_image).save(self.vignette_path)
        self.log_callback("log", "     ✅ Vignette created.")