# Distributed Face Recognition System Setup

## Overview

This guide explains how to set up the face recognition system in a distributed architecture where:

- The face recognition processing runs on a PC with a camera
- The web interface can be hosted on a separate computer

This setup allows you to have the computationally intensive face recognition running on a dedicated machine while making the web interface accessible from anywhere on your network.

## System Architecture

```
+----------------+                 +----------------+
|                |                 |                |
| Recognition PC |<--------------->| Web Server PC  |
| (with camera)  |   TCP/IP        | (Flask app)    |
|                |   Socket        |                |
+----------------+                 +----------------+
                                         ^
                                         |
                                         v
                                   +----------------+
                                   |                |
                                   | Client Browser |
                                   |                |
                                   +----------------+
```

## Setup Instructions

### Step 1: Configure the System

1. Edit the `config.py` file to set the correct IP address:

```python
# Set this to the IP address of the computer running the face_recognition_server.py
RECOGNITION_SERVER_HOST = '192.168.1.X'  # Replace with actual IP of the PC with camera
RECOGNITION_SERVER_PORT = 9999
```

### Step 2: Setup the Recognition Server (PC with Camera)

1. Install all required dependencies:

```
pip install -r requirements.txt
```

2. Run the face recognition server:

```
python face_recognition_server.py
```

This will start the server and initialize the camera. You should see output indicating the server is running and listening for connections.

### Step 3: Setup the Web Server (Any PC)

1. Install all required dependencies:

```
pip install -r requirements.txt
```

2. Make sure the `config.py` file has the correct IP address of the recognition server.

3. Run the Flask web application:

```
python app.py
```

This will start the web server on port 8081 and attempt to connect to the recognition server.

### Step 4: Access the Web Interface

1. Open a web browser on any device on the network
2. Navigate to `http://[WEB_SERVER_IP]:8081`
3. Log in and use the system as normal

## Troubleshooting

### Connection Issues

If the web server cannot connect to the recognition server:

1. Verify both machines are on the same network
2. Check that the IP address in `config.py` is correct
3. Ensure no firewall is blocking port 9999
4. Verify the recognition server is running

### Camera Issues

If the camera is not working:

1. Make sure the camera is properly connected to the recognition server PC
2. Check that no other application is using the camera
3. Restart the recognition server

## Security Considerations

This setup is designed for use within a trusted local network. For additional security:

1. Consider implementing authentication for the socket connection
2. Use a VPN if accessing from outside the local network
3. Set up firewall rules to restrict access to the necessary ports

## Data Flow

1. The recognition server captures video frames from the camera
2. Face recognition is performed on these frames
3. Processed frames and recognition data are sent to connected web servers
4. The web server displays the video feed and marks attendance based on recognized faces
5. User data and photos are sent from the web server to the recognition server for training