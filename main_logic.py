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
import time
import concurrent.futures
try:
    import torch
except Exception:
    torch = None

from caption_styles import CaptionGenerator
from transcriber import Transcriber
from analysis import ContentAnalyzer
from ultralytics import YOLO

DEFAULT_TARGET_RESOLUTION = (1080, 1920)

class YouTubeClipperCore:
    def __init__(self, log_callback, target_resolution=DEFAULT_TARGET_RESOLUTION):
        self.log_callback = log_callback
        self.TARGET_RESOLUTION = target_resolution
        self._internal_message_queue = queue.Queue()

        # BOOST mode: enable aggressive speed tuning (can be forced with AI_CLIPS_FORCE_BOOST=1/0)
        self.boost_mode = os.environ.get("AI_CLIPS_FORCE_BOOST", "1") == "1"
        self.log_callback("log", f"⚡ BOOST mode: {'ENABLED' if self.boost_mode else 'disabled'}")

        # detect hardware accel / encoder early
        self.hw_accel_info = self._detect_hw_accel()
        self.log_callback("log", f"🧠 HW accel: {self.hw_accel_info.get('encoder', 'cpu')}")

        self.caption_generator = CaptionGenerator(self._internal_message_queue)
        self.transcriber = None
        self.analyzer = None
        self.vignette_path = None
        self.final_clip_metadata = []
        self._video_probe_cache = {}  # cache ffprobe results
        
        # Keep YOLO initialization (optional heavy op - can be deferred if needed)
        self.log_callback("log", "🧠 Initializing YOLO Object Detection Model...")
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            self.log_callback("log", "     ✅ YOLO model initialized successfully.")
        except Exception as e:
            self.yolo_model = None
            self.log_callback("log", f"     ⚠️ YOLO initialization failed (will continue): {e}")

        threading.Thread(target=self._process_internal_queue, daemon=True).start()

    def _get_transcriber(self):
        if self.transcriber is None:
            self.log_callback("log", "🧠 Initializing AI Transcription Engine (Whisper)...")
            self.transcriber = Transcriber(self._internal_message_queue)
        return self.transcriber

    def _get_analyzer(self):
        if self.analyzer is None:
            self.log_callback("log", "🧠 Initializing AI Content Analyzer...")
            self.analyzer = ContentAnalyzer(self._internal_message_queue)
        return self.analyzer
    
    def _process_internal_queue(self):
        while True:
            try:
                item = self._internal_message_queue.get()
                msg_type, value = (item[0], item[1] if len(item) > 1 else None)
                self.log_callback(msg_type, value)
            except Exception as e:
                self.log_callback("log", f"ERROR processing internal queue: {e}")
            finally:
                self._internal_message_queue.task_done()

    def _safe_delete(self, file_path, retries=3, delay=1):
        """Attempts to delete a file, with retries for file lock errors."""
        for i in range(retries):
            try:
                if file_path and file_path.exists():
                    file_path.unlink()
                break # Success
            except PermissionError as e:
                if i < retries - 1:
                    self.log_callback("log", f"     ⚠️ File '{file_path.name}' is locked. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self.log_callback("log", f"     ❌ Could not delete locked file '{file_path.name}': {e}")
            except Exception as e:
                self.log_callback("log", f"     ❌ Error deleting file '{file_path.name}': {e}")
                break

    def _intelligent_clipping_pipeline(self, url, num_clips, clip_duration_arg, add_captions, font_choice, caption_style, download_folder_path, temp_folder_path):
        # Normalize paths and provide a fallback temp folder when caller didn't pass one
        download_folder_path = Path(download_folder_path)
        temp_folder_path = Path(temp_folder_path)

        # ensure folders exist
        download_folder_path.mkdir(parents=True, exist_ok=True)
        temp_folder_path.mkdir(parents=True, exist_ok=True)

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
            # --- Step 1: Download the video file (with audio) ---
            self.log_callback("log", "🚀 Downloading full video for analysis...")
            self.log_callback("progress", 10)
            
            # This format string is more flexible
            ydl_video_opts = {
                'format': 'bestvideo+bestaudio/best',
                'outtmpl': str(download_folder_path / 'full_source_video.%(ext)s'),
                'merge_output_format': 'mp4'
            }

            with YoutubeDL(ydl_video_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # After download, the filename will be .mp4 because of 'merge_output_format'
                downloaded_filepath = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp4'


            full_video_path = Path(downloaded_filepath) if Path(downloaded_filepath).exists() else None
            if not full_video_path: raise FileNotFoundError("Downloaded video file not found.")
            self.log_callback("log", f"✅ Full video downloaded: {full_video_path.name}")
            self.log_callback("progress", 20)
            
            self._create_source_thumbnail(full_video_path, download_folder_path)

            # --- Step 2: Extract audio as WAV for Whisper ---
            self.log_callback("log", "🔊 Extracting audio for Whisper...")
            self.log_callback("progress", 25)
            full_audio_path = download_folder_path / 'full_source_audio.wav'
            
            subprocess.run(["ffmpeg", "-i", str(full_video_path), "-y", "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(full_audio_path)], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            if not full_audio_path.exists(): raise FileNotFoundError("Extracted audio file not found.")
            self.log_callback("progress", 30)

            # --- Continue with the rest of the pipeline ---
            timed_words = self._get_transcriber().get_word_timestamps(str(full_audio_path))
            if timed_words is None:
                raise Exception("Transcription failed. Please check AI model or audio file.")
            self.log_callback("progress", 40)

            best_clips = self._get_analyzer().find_best_clips(str(full_audio_path), timed_words, num_clips, clip_duration_arg)
            if not best_clips:
                self.log_callback("log", "Could not find any suitable clips based on AI analysis.")
                return True 
            
            # Parallelize clip processing
            if self.boost_mode and ("nvenc" in (self.hw_accel_info.get("encoder") or "") or "vaapi" in (self.hw_accel_info.get("encoder") or "")):
                max_workers = min(len(best_clips), max(1, self._gpu_count()))
            else:
                max_workers = min(len(best_clips), max(1, (os.cpu_count() or 2)))
            
            self.log_callback("log", f"⚡ Processing {len(best_clips)} clips with {max_workers} workers...")
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i, clip_info in enumerate(best_clips):
                    futures.append(ex.submit(self._process_clip_task, i, clip_info, add_captions, font_choice, caption_style, timed_words, full_video_path, download_folder_path, temp_folder_path))
                for fut in concurrent.futures.as_completed(futures):
                    ok, metadata = fut.result()
                    if ok and metadata:
                        self.final_clip_metadata.append(metadata)

            self.log_callback("log", "\n✅ All clips generated successfully!")
            return True
        except Exception as e:
            import traceback
            self.log_callback("log", f"❌ A critical error occurred in the pipeline: {e}\n{traceback.format_exc()}")
            return False
        finally:
            self._safe_delete(full_video_path)
            self._safe_delete(full_audio_path)
            
            if temp_folder_path.exists():
                try:
                    shutil.rmtree(temp_folder_path)
                except OSError as e:
                    self.log_callback("log", f"⚠️ Could not remove temp folder {temp_folder_path}: {e}")
            
            self.log_callback("log", "--- Core processing pipeline finished ---")
            
    def _create_source_thumbnail(self, video_path, output_folder):
        self.log_callback("log", "     📸 Generating source thumbnail...")
        thumbnail_path = output_folder / "source_thumbnail.jpg"
        try:
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            command = ["ffmpeg", "-y", "-ss", "00:00:01", "-i", str(video_path), "-vframes", "1", "-q:v", "2", str(thumbnail_path)]
            subprocess.run(command, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except (subprocess.CalledProcessError, Exception) as e:
            self.log_callback("log", f"     ❌ Failed to generate source thumbnail: {e}")

    def _process_clip_task(self, i, clip_info, add_captions, font_choice, caption_style, timed_words, full_video_path, download_folder_path, temp_folder_path):
        """
        Worker task to process a single clip.
        This now creates one clip with three internal styles.
        """
        try:
            self.log_callback("log", f"\n--- Processing Clip {i+1} (worker) ---")
            clip_words = [word.copy() for word in timed_words if clip_info['start'] <= word['start'] < clip_info['end']]
            for word in clip_words:
                word['start'] -= clip_info['start']
                word['end'] -= clip_info['start']

            temp_output_filename = f"clip_{i+1}_raw_output.mp4"

            # The main call to the FFmpeg processor.
            self._process_single_clip_with_ffmpeg(
                str(full_video_path),
                clip_info,
                temp_output_filename,
                add_captions,
                font_choice,
                caption_style,
                clip_words,
                download_folder_path,
                temp_folder_path
            )

            temp_path = download_folder_path / temp_output_filename
            # validate that ffmpeg produced a playable file (duration > 0)
            if temp_path.exists():
                try:
                    duration_str = subprocess.check_output([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", str(temp_path)
                    ]).decode('utf-8').strip()
                    dur = float(duration_str) if duration_str else 0.0
                except Exception:
                    dur = 0.0
                if dur <= 0:
                    self.log_callback("log", f"     ⚠️ Generated clip has zero duration ({temp_path}). Treating as failed.")
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                    return False, None
                meta = self._get_analyzer().generate_title_and_description(clip_words)
                sanitized_title = re.sub(r'[\\/*?:"<>|]', "", meta.get('title', f"clip_{i+1}"))
                
                # Truncate the sanitized title to a safe length (e.g., 30 characters)
                if len(sanitized_title) > 30:
                    sanitized_title = sanitized_title[:30].strip()
                
                final_filename = f"final_clip_{i+1}_{sanitized_title}.mp4"

                # Determine canonical output folder under downloaded_clips so the
                # dashboard (which reads downloaded_clips/) will see generated clips.
                repo_root = Path.cwd()
                # prefer using the session folder name from download_folder_path; fall back to temp folder name
                session_id = download_folder_path.name or temp_folder_path.name
                canonical_out_dir = repo_root / 'clips' / session_id
                canonical_out_dir.mkdir(parents=True, exist_ok=True)

                final_path = canonical_out_dir / final_filename

                # Create a thumbnail for the final composed clip inside the canonical folder
                self._create_video_thumbnail(temp_path, canonical_out_dir, final_filename.replace('.mp4', '_thumb.jpg'))

                try:
                    # Use shutil.move to handle cross-filesystem moves robustly
                    shutil.move(str(temp_path), str(final_path))
                    meta['filename'] = final_filename
                    return True, meta
                except OSError as e:
                    self.log_callback("log", f"     ❌ Failed to move {temp_output_filename} to downloaded_clips: {e}")
                    meta['filename'] = temp_output_filename
                    return True, meta
            else:
                self.log_callback("log", f"     ⚠️ Expected output not found: {temp_output_filename}")
                return False, None

        except Exception as e:
            import traceback
            self.log_callback("log", f"     ❌ Worker failed for clip {i+1}: {e}\n{traceback.format_exc()}")
            return False, None
    
    def _create_video_thumbnail(self, video_path, output_folder, thumb_filename="thumbnail.jpg"):
        """Generates a thumbnail for a given video file."""
        self.log_callback("log", f"     📸 Generating thumbnail for {video_path.name}...")
        thumbnail_path = output_folder / thumb_filename
        try:
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            # Get a frame from 1 second in, or the middle if the clip is short
            duration_str = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
            ).decode('utf-8').strip()
            duration = float(duration_str)
            thumb_time = "00:00:01" if duration > 2 else f"00:00:0{duration / 2:.2f}"

            command = ["ffmpeg", "-y", "-ss", thumb_time, "-i", str(video_path), "-vframes", "1", "-q:v", "3", str(thumbnail_path)]
            subprocess.run(command, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception as e:
            self.log_callback("log", f"     ❌ Failed to generate video thumbnail: {e}")


    def _process_single_clip_with_ffmpeg(self, source_video_path, clip_info, output_filename, add_captions, font_choice, caption_style, timed_words, download_folder_path, temp_folder_path):
        """
        Builds and runs an FFmpeg command that composes a multi-style clip.
        - Part 1: Blurred background with a centered foreground.
        - Part 2: Split-screen effect over a blurred background.
        - Part 3: Crop-to-fill effect over a blurred background.
        """
        start_time = clip_info['start']
        duration = clip_info['end'] - clip_info['start']

        self.log_callback("log", "     🔥 Building multi-style FFmpeg command...")

        encoder = self.hw_accel_info.get("encoder", "libx264")
        hw_args = list(self.hw_accel_info.get("hwaccel_args", []))

        # Detect CUDA availability on Windows; if missing, avoid adding '-hwaccel cuda'
        # and fallback from nvenc to libx264 to ensure ffmpeg can decode frames for filters.
        try:
            from pathlib import Path
            cuda_available = False
            if os.name == "nt":
                cuda_available = Path(r"C:\Windows\System32\nvcuda.dll").exists()
            if not cuda_available and any("cuda" in str(a).lower() for a in hw_args):
                self.log_callback("log", "⚠️ CUDA requested but nvcuda.dll not found — falling back to software decode/encode.")
                hw_args = []
                if "nvenc" in (encoder or "").lower():
                    encoder = "libx264"
        except Exception:
            hw_args = hw_args

        if "nvenc" in encoder:
            codec_args = ["-c:v", encoder, "-preset", "p1" if self.boost_mode else "p4", "-rc", "vbr", "-cq", "24" if self.boost_mode else "20", "-b:v", "8000k"]
        else:
            codec_args = ["-c:v", "libx264", "-preset", "superfast" if self.boost_mode else "veryfast", "-crf", "20" if self.boost_mode else "18", "-b:v", "6000k" if self.boost_mode else "8000k"]

        # Build base ffmpeg command
        ffmpeg_cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "warning"] + hw_args + ["-ss", str(start_time), "-t", str(duration), "-i", source_video_path, "-threads", "0"]

        # Segment durations
        seg_dur = duration / 3.0
        seg0 = seg_dur
        seg1 = seg_dur
        seg2 = max(0.0, duration - seg0 - seg1)
        fade_dur = 0.5

        W, H = self.TARGET_RESOLUTION

        # Build filter graph with constant frame rate trims to satisfy xfade
        graph_lines = []
        TARGET_FPS = 30
        graph_lines.append(f"[0:v]trim=start=0:duration={seg0},setpts=PTS-STARTPTS,fps={TARGET_FPS}[v0];")
        graph_lines.append(f"[0:v]trim=start={seg0}:duration={seg1},setpts=PTS-STARTPTS,fps={TARGET_FPS}[v1];")
        graph_lines.append(f"[0:v]trim=start={seg0 + seg1}:duration={seg2},setpts=PTS-STARTPTS,fps={TARGET_FPS}[v2];")

        # Segment 0: blurred background + centered foreground
        fg0_h = int(H * 0.78)
        graph_lines.append(f"[v0]scale={W}:{H},boxblur=15:5,eq=brightness=0.03:saturation=0.95[bg0];")
        graph_lines.append(f"[v0]scale=-2:{fg0_h},format=rgba[fg0];")
        graph_lines.append(f"[bg0][fg0]overlay=(W-w)/2:(H-h)/2,format=yuv420p,setsar=1[s0];")

        # Segment 1: split/stack - MODIFIED FOR VERTICAL STACKING
        half_h = int(H // 2)
        graph_lines.append(f"[v1]scale={W}:{half_h}[t1];")
        graph_lines.append(f"[v1]scale={W}:{half_h}[b1];")
        graph_lines.append(f"[t1][b1]vstack=inputs=2,scale={W}:{H},format=yuv420p[s1];")

        # Segment 2: fill/crop
        graph_lines.append(f"[v2]scale='if(gt(a,{W}/{H}),-2,{W})':'if(gt(a,{W}/{H}),{H},-2)',crop={W}:{H},format=yuv420p[s2];")

        # Ensure s0/s1/s2 have same fps/format (defensive)
        graph_lines.append(f"[s0]fps={TARGET_FPS},format=yuv420p[s0f];")
        graph_lines.append(f"[s1]fps={TARGET_FPS},format=yuv420p[s1f];")
        graph_lines.append(f"[s2]fps={TARGET_FPS},format=yuv420p[s2f];")

        # XFADE chain
        off1 = max(0.0, seg0 - fade_dur)
        off2 = max(0.0, seg0 + seg1 - fade_dur)
        graph_lines.append(f"[s0f][s1f]xfade=transition=fade:duration={fade_dur}:offset={off1}[xf1];")
        graph_lines.append(f"[xf1][s2f]xfade=transition=fade:duration={fade_dur}:offset={off2}[vout];")

        # Optional captions
        caption_images = []
        if add_captions and timed_words:
            caption_images = self.caption_generator.create_caption_images(timed_words, font_choice, caption_style, temp_folder_path)

        if caption_images:
            # append caption image inputs
            for img in caption_images:
                if img.get("path"):
                    ffmpeg_cmd.extend(["-i", str(img["path"])])

            last = "[vout]"
            caption_start_idx = 1
            for idx, img_info in enumerate(caption_images):
                stream_idx = idx + caption_start_idx
                start_cap = round(float(img_info.get('start', 0.0)), 3)
                end_cap = round(float(img_info.get('end', seg_dur)), 3)
                overlay_filter = f"overlay=(W-w)/2:(H-h)*0.7:enable='between(t,{start_cap},{end_cap})'"
                out_label = f"[vcap{idx}]"
                graph_lines.append(f"{last}[{stream_idx}:v]{overlay_filter}{out_label};")
                last = out_label
            graph_lines.append(f"{last}format=yuv420p[vfinal];")
            final_label = "[vfinal]"
        else:
            final_label = "[vout]"

        # assemble filter_complex
        filter_complex_graph = "".join(line + "\n" for line in graph_lines)

        # prepare mapping and final cmd
        final_map_stream = final_label.replace('[', '').replace(']', '')
        self.log_callback("log", f"     🔍 FFmpeg filter_complex graph:\n{filter_complex_graph}")
        ffmpeg_cmd.extend(["-filter_complex", filter_complex_graph, "-map", f"[{final_map_stream}]", "-map", "0:a"]) 
        ffmpeg_cmd.extend(codec_args)
        ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "160k", "-pix_fmt", "yuv420p"]) 
        output_path = str(download_folder_path / output_filename)
        # Use a temporary filename with a recognizable .mp4 extension so ffmpeg
        # can choose the correct muxer. We'll atomically rename it to the final
        # name after a successful run.
        tmp_output = output_path + ".part.mp4"
        # enable faststart for better streaming playback
        ffmpeg_cmd.extend(["-movflags", "+faststart"])
        ffmpeg_cmd.append(tmp_output)

        try:
            self.log_callback("log", "     🚀 Executing FFmpeg multi-style command...")
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            # move partial to final atomically
            try:
                if Path(tmp_output).exists():
                    Path(tmp_output).rename(output_path)
            except Exception as e:
                self.log_callback("log", f"     ⚠️ Could not rename temp output: {e}")
            self.log_callback("log", "     ✅ FFmpeg command completed successfully.")
        except subprocess.CalledProcessError as e:
            self.log_callback("log", f"     ❌ FFmpeg Error: {e.stderr}")
            # clean up partial output if present
            try:
                if Path(tmp_output).exists():
                    Path(tmp_output).unlink()
            except Exception:
                pass
            raise
        finally:
            for img_info in caption_images:
                try:
                    self._safe_delete(Path(img_info.get('path')))
                except Exception:
                    pass

    def _detect_hw_accel(self):
        """Detects available hardware acceleration and encoders."""
        # This is a placeholder for a more robust detection mechanism.
        # For now, it prioritizes NVIDIA, then falls back to CPU.
        try:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
            if "h264_nvenc" in result.stdout:
                return {
                    "encoder": "h264_nvenc",
                    "hwaccel_args": ["-hwaccel", "cuda"]
                }
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return {"encoder": "libx264", "hwaccel_args": []}

    def _gpu_count(self):
        # Placeholder for detecting number of GPUs, useful for NVIDIA
        return 1

    def _cache_video_probe(self, video_path):
        """Runs ffprobe and caches the result."""
        if str(video_path) in self._video_probe_cache:
            return self._video_probe_cache[str(video_path)]
        try:
            probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            self._video_probe_cache[str(video_path)] = json.loads(result.stdout)
            return self._video_probe_cache[str(video_path)]
        except Exception as e:
            self.log_callback("log", f"     ⚠️ ffprobe failed for {video_path.name}: {e}")
            return None