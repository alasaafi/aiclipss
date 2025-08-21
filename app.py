# app.py

import subprocess
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import threading
from pathlib import Path
import time
import shutil
import json
import traceback
from datetime import date, datetime, timedelta
from yt_dlp import YoutubeDL

# Import your business logic class
from main_logic import YouTubeClipperCore

app = Flask(__name__)

# --- Configuration for Database and Login ---
app.config['SECRET_KEY'] = 'a-super-secret-key-that-you-should-change'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Folder Configuration ---
DOWNLOAD_FOLDER = Path("clips")
TEMP_FOLDER = Path("temp")
LOGS_FOLDER = Path("logs")

for folder in [DOWNLOAD_FOLDER, TEMP_FOLDER, LOGS_FOLDER]:
    folder.mkdir(exist_ok=True)
os.environ["PATH"] += os.pathsep + r"C:\\ffmpeg\\bin"
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['LOGS_FOLDER'] = LOGS_FOLDER

# --- Initialize Extensions ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Custom handler for unauthorized requests
@login_manager.unauthorized_handler
def unauthorized():
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify(error="Authentication required.", message="Please log in to continue."), 401
    return redirect(url_for('login'))


# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    tier = db.Column(db.String(50), default='free', nullable=False)
    clip_count_today = db.Column(db.Integer, default=0, nullable=False)
    last_clip_date = db.Column(db.Date, default=date.today(), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    processes = db.relationship('VideoProcess', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class VideoProcess(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    clips = db.relationship('Clip', backref='process', lazy=True, cascade="all, delete-orphan")
    source_url = db.Column(db.String(500))
    status = db.Column(db.String(20), default='processing', nullable=False)

class Clip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    title = db.Column(db.String(300), nullable=False)
    description = db.Column(db.Text, nullable=False)
    duration = db.Column(db.Float, default=0.0)
    process_id = db.Column(db.Integer, db.ForeignKey('video_process.id'), nullable=False)

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Custom Decorator for Admin Access ---
from functools import wraps

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Page Routes ---
@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/health')
def health_check():
    """A simple health check endpoint for Render's health monitoring."""
    return "OK", 200

# --- Authentication Routes ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        
        user_by_email = User.query.filter_by(email=email).first()
        if user_by_email:
            flash('Email already registered.', 'error')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email, tier='free', clip_count_today=0, last_clip_date=date.today())
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Please check your login details and try again.', 'error')
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Dashboard Route ---
@app.route('/dashboard')
@login_required
def dashboard():
    user_processes = VideoProcess.query.filter_by(user_id=current_user.id).order_by(VideoProcess.created_at.desc()).all()
    
    live_session_id = session.get('processing_session_id')
    live_process = None
    live_process_thumbnail_url = None

    if live_session_id:
        live_process = VideoProcess.query.filter_by(session_id=live_session_id, user_id=current_user.id).first()
        if live_process:
            source_thumbnail_path = Path(app.config['DOWNLOAD_FOLDER']) / live_session_id / 'source_thumbnail.jpg'
            if source_thumbnail_path.exists():
                live_process_thumbnail_url = url_for('serve_thumbnail', session_id=live_session_id, filename='source_thumbnail.jpg')

        if live_process and live_process.status != 'processing':
            session.pop('processing_session_id', None)
            live_process = None
    
    notification_count = 3 # Placeholder
    has_notifications = notification_count > 0

    return render_template(
        'dashboard.html', 
        processes=user_processes, 
        current_user_tier=current_user.tier, 
        current_clip_count=current_user.clip_count_today,
        live_process=live_process,
        live_session_id=live_session_id,
        live_process_thumbnail_url=live_process_thumbnail_url,
        notification_count=notification_count,
        has_notifications=has_notifications
    )

@app.route('/get_video_info')
def get_video_info():
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "URL parameter is missing."}), 400

    try:
        ydl_opts = {
            'format': 'bestvideo',
            'quiet': True,
            'skip_download': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            height = info['height']
            resolution = f"{height}p"
            return jsonify({"resolution": resolution})
    except Exception as e:
        app.logger.error(f"Failed to get video info for {url}: {e}")
        return jsonify({"error": "Failed to retrieve video information."}), 500

# --- Video Processing Route ---
@app.route('/process_video', methods=['POST'])
@login_required
def process_video_request():
    try:
        # --- Tier-based Feature Gating ---
        FREE_TIER_CLIP_LIMIT = 10
        FREE_TIER_MAX_DURATION = 90
        FREE_TIER_RESOLUTION = (720, 1280)  # 720p Vertical
        PRO_TIER_RESOLUTION = (1080, 1920) # 1080p Full HD Vertical
        BUSINESS_TIER_RESOLUTION = (1440, 2560) # 2K QHD Vertical

        if current_user.tier == 'free':
            if current_user.last_clip_date != date.today():
                current_user.clip_count_today = 0
                current_user.last_clip_date = date.today()
                db.session.commit()
            
            if current_user.clip_count_today >= FREE_TIER_CLIP_LIMIT:
                return jsonify({"status": "error", "message": f"Free tier limit reached. You can only generate {FREE_TIER_CLIP_LIMIT} clips per day."}), 403

        data = request.form
        url = data.get('youtube_url')
        clip_length_str = data.get('clip_length')
        num_clips_str = data.get('num_clips')
        add_captions = data.get('add_captions') == 'on'
        font_choice_raw = data.get('font_choice')
        
        font_choice_mapping = {
            "Gabarito": ("Gabarito", "Karaoke"),
            "Anton": ("Anton", "Karaoke"),
            "Montserrat": ("Montserrat", "Opus Green/White"),
            "Montserrat-Blue": ("Montserrat", "Opus Blue/White")
        }
        
        font_choice, caption_style = font_choice_mapping.get(font_choice_raw, ("Gabarito", "Karaoke"))

        if not url:
            return jsonify({"status": "error", "message": "Please enter a YouTube URL."}), 400

        try:
            num_clips = int(num_clips_str)
            clip_duration_arg = int(clip_length_str)

            if current_user.tier == 'free':
                if (current_user.clip_count_today + num_clips) > FREE_TIER_CLIP_LIMIT:
                    return jsonify({"status": "error", "message": f"Generating this many clips would exceed your free tier limit of {FREE_TIER_CLIP_LIMIT} per day."}), 403
                if clip_duration_arg > FREE_TIER_MAX_DURATION:
                    return jsonify({"status": "error", "message": f"Free tier clips are limited to {FREE_TIER_MAX_DURATION} seconds."}), 403

            if not (1 <= num_clips <= 10) or not (15 <= clip_duration_arg <= 180):
                raise ValueError("Clip settings are outside the allowed range.")

        except (ValueError, TypeError) as e:
            app.logger.error(f"Invalid form input in /process_video: {e}")
            return jsonify({"status": "error", "message": "Invalid input for number of clips or clip length."}), 400

        session_id = str(time.time()).replace('.', '')

        new_process = VideoProcess(
            session_id=session_id, 
            user_id=current_user.id,
            source_url=url,
            status='processing'
        )
        db.session.add(new_process)
        
        if current_user.tier == 'free':
            current_user.clip_count_today += num_clips
        db.session.commit()
        
        session['processing_session_id'] = session_id

        process_db_id = new_process.id
        
        session_download_folder = DOWNLOAD_FOLDER / session_id
        session_temp_folder = TEMP_FOLDER / session_id
        session_download_folder.mkdir(exist_ok=True, parents=True)
        session_temp_folder.mkdir(exist_ok=True, parents=True)
        log_file_path = LOGS_FOLDER / f"log_{session_id}.txt"

        target_resolution = PRO_TIER_RESOLUTION
        if current_user.tier == 'free':
            target_resolution = FREE_TIER_RESOLUTION
        elif current_user.tier == 'business':
            target_resolution = BUSINESS_TIER_RESOLUTION

        args = (
            app, process_db_id, url, num_clips, clip_duration_arg,
            add_captions, font_choice, caption_style,
            session_download_folder, session_temp_folder,
            log_file_path, target_resolution
        )

        thread = threading.Thread(target=run_clipping_pipeline, args=args, daemon=True)
        thread.start()

        return jsonify({"status": "processing", "redirect_url": url_for('dashboard')}), 202
    except Exception as e:
        app.logger.error(f"An unhandled error occurred in /process_video: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500

# --- Clipping Pipeline Runner ---
def run_clipping_pipeline(flask_app, process_db_id, url, num_clips, clip_duration_arg, add_captions, font_choice, caption_style, session_download_folder, session_temp_folder, log_file_path, target_resolution):
    with flask_app.app_context():
        # Use the existing Flask-SQLAlchemy session
        session = db.session

        def log_to_file(message_type, value):
            with open(log_file_path, 'a', encoding='utf-8') as f:
                if message_type == "log":
                    f.write(f"{value}\n")
                elif message_type == "progress":
                    f.write(f"PROGRESS:{value}\n")

        clipper_core = YouTubeClipperCore(log_to_file, target_resolution=target_resolution) 
        
        try:
            clipper_core._intelligent_clipping_pipeline(
                url, num_clips, clip_duration_arg, add_captions, font_choice, caption_style,
                session_download_folder, session_temp_folder
            )
            
            process_entry = session.get(VideoProcess, process_db_id)
            if process_entry:
                for clip_meta in clipper_core.final_clip_metadata:
                    # Get clip duration
                    clip_path = session_download_folder / clip_meta['filename']
                    try:
                        duration_str = subprocess.check_output([
                            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(clip_path)
                        ]).decode('utf-8').strip()
                        clip_duration = float(duration_str) if duration_str else 0.0
                    except Exception as e:
                        print(f"Error probing duration for {clip_meta['filename']}: {e}")
                        clip_duration = 0.0
                        
                    new_clip = Clip(
                        filename=clip_meta['filename'],
                        title=clip_meta['title'],
                        description=clip_meta['description'],
                        duration=clip_duration,
                        process_id=process_entry.id
                    )
                    session.add(new_clip)
                
                process_entry.status = 'completed'
                session.commit()
            
            final_metadata_path = session_download_folder / "metadata.json"
            with open(final_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(clipper_core.final_clip_metadata, f, indent=4)
                
            log_to_file("log", "PROCESSING_COMPLETE: All clips generated and metadata saved to dashboard.")
        except Exception as e:
            log_to_file("log", f"PROCESSING_ERROR: {e}\n{traceback.format_exc()}")
            process_entry = session.get(VideoProcess, process_db_id)
            if process_entry:
                process_entry.status = 'error'
            session.rollback()
        finally:
            session.remove()

# --- New: Delete Process Route ---
@app.route('/delete_process/<session_id>', methods=['DELETE'])
@login_required
def delete_process(session_id):
    try:
        process_to_delete = VideoProcess.query.filter_by(session_id=session_id, user_id=current_user.id).first()
        
        if not process_to_delete:
            return jsonify({"status": "error", "message": "Project not found or you don't have permission to delete it."}), 404

        db.session.delete(process_to_delete)
        db.session.commit()

        session_download_folder = DOWNLOAD_FOLDER / session_id
        session_temp_folder = TEMP_FOLDER / session_id
        log_file_path = LOGS_FOLDER / f"log_{session_id}.txt"

        for folder_path in [session_download_folder, session_temp_folder]:
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    app.logger.info(f"Deleted folder: {folder_path}")
                except OSError as e:
                    app.logger.error(f"Error deleting folder {folder_path}: {e}")
        
        if log_file_path.exists():
            try:
                log_file_path.unlink()
                app.logger.info(f"Deleted log file: {log_file_path}")
            except OSError as e:
                app.logger.error(f"Error deleting log file {log_file_path}: {e}")
                
        if session.get('processing_session_id') == session_id:
            session.pop('processing_session_id', None)

        flash('Project and all associated clips deleted successfully!', 'success')
        return jsonify({"status": "success", "message": "Project deleted."}), 200

    except Exception as e:
        app.logger.error(f"Error deleting process {session_id}: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "Failed to delete project due to an internal server error."}), 500


# --- Tier Purchase Placeholder Route ---
@app.route('/purchase/<tier_name>', methods=['POST'])
@login_required
def purchase_tier(tier_name):
    if tier_name not in ['pro', 'business']:
        flash('Invalid tier selected.', 'error')
        return jsonify({"status": "error", "message": "Invalid tier."}), 400
    
    if tier_name == 'pro':
        return jsonify({
            "status": "redirect",
            "message": "Redirecting to Gumroad to complete your purchase.",
            "redirect_url": "https://aiclips.gumroad.com/l/jxjgx"
        })
    
    current_user.tier = tier_name
    current_user.clip_count_today = 0
    current_user.last_clip_date = date.today()
    db.session.commit()

    flash(f'Successfully upgraded to {tier_name.upper()} plan!', 'success')
    return jsonify({"status": "success", "message": f"Upgraded to {tier_name} plan."}), 200

# --- Webhook endpoint to handle Gumroad purchases ---
@app.route('/gumroad-webhook', methods=['POST'])
def gumroad_webhook():
    data = request.form
    product_id = data.get('product_id')
    customer_email = data.get('email')

    PRO_PLAN_GUMROAD_PRODUCT_ID = "jxjgx" 

    if product_id != PRO_PLAN_GUMROAD_PRODUCT_ID:
        return jsonify({"message": "Product ID mismatch"}), 400

    if not customer_email:
        return jsonify({"message": "Missing customer email"}), 400

    user_to_upgrade = User.query.filter_by(email=customer_email).first()

    if user_to_upgrade:
        user_to_upgrade.tier = 'pro'
        user_to_upgrade.clip_count_today = 0
        user_to_upgrade.last_clip_date = date.today()
        db.session.commit()
        app.logger.info(f"User {user_to_upgrade.username} upgraded to Pro via Gumroad.")
        return jsonify({"status": "success", "message": "User plan upgraded"}), 200
    else:
        app.logger.warning(f"Gumroad webhook received for unknown email: {customer_email}")
        return jsonify({"status": "error", "message": "User not found"}), 404

# --- Admin Routes ---
@app.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template('admin/dashboard.html', users=users)

@app.route('/admin/gift_plan', methods=['POST'])
@admin_required
def admin_gift_plan():
    username = request.form.get('username')
    new_tier = request.form.get('new_tier')

    if not username or new_tier not in ['free', 'pro', 'business']:
        flash('Invalid input for gifting plan.', 'error')
        return redirect(url_for('admin_dashboard'))

    user_to_update = User.query.filter_by(username=username).first()
    if not user_to_update:
        flash(f'User "{username}" not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    user_to_update.tier = new_tier
    user_to_update.clip_count_today = 0
    user_to_update.last_clip_date = date.today()
    db.session.commit()
    
    flash(f'Successfully gifted {new_tier.upper()} plan to {username}!', 'success')
    return redirect(url_for('admin_dashboard'))


# --- Utility Routes ---
@app.route('/logs/<session_id>')
@login_required
def stream_logs(session_id):
    def generate():
        log_file_path = LOGS_FOLDER / f"log_{session_id}.txt"
        last_position = 0
        processing_complete = False
        
        while not log_file_path.exists():
            time.sleep(0.5)
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            while not processing_complete:
                f.seek(last_position)
                new_content = f.read()
                
                if new_content:
                    lines = new_content.splitlines()
                    for line in lines:
                        yield f"data: {line}\n\n"
                        if "PROCESSING_COMPLETE:" in line or "PROCESSING_ERROR:" in line:
                            processing_complete = True
                    last_position = f.tell()
                
                if not processing_complete:
                    time.sleep(0.5)
        
        yield "data: STREAM_TERMINATED\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/get_clips_metadata/<session_id>')
@login_required
def get_clips_metadata(session_id):
    metadata_file = DOWNLOAD_FOLDER / session_id / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return jsonify({"status": "success", "metadata": metadata})
    else:
        return jsonify({"status": "error", "message": "Processing not finished or metadata not found."}), 404

@app.route('/download/<session_id>/<filename>')
@login_required
def download_clip(session_id, filename):
    return send_from_directory(DOWNLOAD_FOLDER / session_id, filename, as_attachment=True)

@app.route('/thumbnail/<session_id>/<filename>')
@login_required
def serve_thumbnail(session_id, filename):
    thumbnail_folder = DOWNLOAD_FOLDER / session_id
    if not thumbnail_folder.exists():
        return jsonify({"status": "error", "message": "Thumbnail not found."}), 404
        
    return send_from_directory(thumbnail_folder, filename, as_attachment=False)

@app.route('/uploader_web/<session_id>')
@login_required
def uploader_web(session_id):
    return render_template('uploader_web.html', session_id=session_id)

@app.route('/perform_upload', methods=['POST'])
@login_required
def perform_upload():
    data = request.json
    return jsonify({"status": "upload_started", "message": "Upload process initiated."})


if __name__ == '__main__':
    pass
    