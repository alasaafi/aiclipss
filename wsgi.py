import os # Import the 'os' module
from waitress import serve
from app import app

if __name__ == '__main__':
    # Get port from environment variable, or default to 10000 for local use
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port) 