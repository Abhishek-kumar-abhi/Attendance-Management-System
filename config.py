# Configuration file for the face recognition system
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server configuration
# Set this to the IP address of the computer running the face_recognition_server.py
# Note: 0.0.0.0 is a binding address for servers, not a connection address for clients
# Use 127.0.0.1 (localhost) for local connections or the actual IP address for remote connections
RECOGNITION_SERVER_HOST = os.environ.get('RECOGNITION_SERVER_HOST', '127.0.0.1')
RECOGNITION_SERVER_PORT = int(os.environ.get('RECOGNITION_SERVER_PORT', 9999))

# Database configuration
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'face_recognition.db')

# Upload configuration
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB max upload size

# Recognition configuration
KNOWN_FACES_DIR = os.environ.get('KNOWN_FACES_DIR', 'known_faces')
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.85))

# Security configuration
DEBUG = os.environ.get('FLASK_ENV', 'production') != 'production'
SSL_ENABLED = os.environ.get('SSL_ENABLED', 'False').lower() == 'true'