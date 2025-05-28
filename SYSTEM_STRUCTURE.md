# Face Recognition System Structure

This document serves as an entry point to understand the structure and architecture of the face recognition system. It provides an overview of the system components and how they interact with each other.

## Documentation Overview

To understand the system architecture, please refer to the following documents:

1. **SYSTEM_ARCHITECTURE.md** - Comprehensive overview of all system components and their interactions
2. **COMPONENT_DIAGRAM.md** - Visual representation of system components and their relationships
3. **DATA_FLOW_DIAGRAM.md** - Detailed illustration of how data moves through the system

## System Components

The face recognition system consists of the following main components:

### Core Components

- **Web Application (app.py)** - Flask-based web interface for users and administrators
- **Face Recognition Server (face_recognition_server.py)** - Handles face recognition processing
- **Face Recognition Client (face_recognition_client.py)** - Connects web app to recognition server
- **Face Recognition Model (Restnet_face_recognition_model.py)** - AI model for face detection and recognition
- **Configuration (config.py)** - Central configuration settings

### Storage Components

- **Database (face_recognition.db)** - SQLite database for user and attendance data
- **Known Faces Directory** - Storage for registered face images and encodings
- **Uploads Directory** - Temporary storage for uploaded images

### Web Interface

- **Templates** - HTML templates for different pages (login, register, dashboard, etc.)
- **Static Assets** - CSS styles and other static resources

## How Components Connect

The system follows a client-server architecture where:

1. The web application provides the user interface and manages database operations
2. The recognition client acts as a bridge between the web app and recognition server
3. The recognition server processes video streams and performs face recognition
4. The face recognition model provides the AI capabilities
5. All components share configuration settings from config.py

## Key Interactions

- **User Registration**: Web app → Database → Recognition server → Known faces storage
- **Authentication**: Web app → Database → Session management
- **Live Recognition**: Camera → Recognition server → Recognition model → Client → Web app
- **Attendance Recording**: Recognition results → Web app → Database

## Deployment Options

The system can be deployed in two main configurations:

1. **Single Machine Deployment** - All components on one machine
2. **Distributed Deployment** - Components distributed across multiple machines

For distributed deployment, the recognition server can run on a machine with GPU capabilities, while the web application can run on a separate web server.

## Configuration

The `config.py` file contains essential settings that affect multiple components:

```python
# Server configuration
RECOGNITION_SERVER_HOST = '0.0.0.0'  # Change to actual server IP
RECOGNITION_SERVER_PORT = 9999

# Database configuration
DATABASE_PATH = 'face_recognition.db'

# Upload configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# Recognition configuration
KNOWN_FACES_DIR = 'known_faces'
CONFIDENCE_THRESHOLD = 0.85
```

For more detailed information about the system architecture and component interactions, please refer to the documentation files mentioned above.