# Stage 1: Build Environment
# Use a slim image for building to keep it as small as possible
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Runtime Environment ---
# Use a minimal base image to reduce final size
FROM python:3.10-alpine

# Set the working directory
WORKDIR /app

# Copy the application code and installed dependencies from the builder stage
COPY --from=builder /app /app

# Copy all other project files needed for runtime
COPY . .

# Set the entry point for your application
CMD ["python", "app.py"]