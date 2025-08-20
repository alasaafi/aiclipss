# === Stage 1: Build Environment ===
# Use a slim Python image for building to keep it as small as possible
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install build-time dependencies including ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Copy the requirements file and install dependencies
# This is where all the large libraries like tensorflow, torch, etc., will be installed
COPY aiclips/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Stage 2: Final Runtime Environment ===
# Use a minimal base image for the final deployment
FROM python:3.10-alpine

# Set the working directory
WORKDIR /app

# Copy the installed packages from the builder stage
# This is the key step that discards all the unnecessary build tools and cache
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the rest of your application code
# This copies your Python scripts, HTML templates, and static files
COPY aiclips/ .

# Ensure FFmpeg is available in the final image
RUN apk add --no-cache ffmpeg

# Set the entry point for your application
CMD ["python", "app.py"]