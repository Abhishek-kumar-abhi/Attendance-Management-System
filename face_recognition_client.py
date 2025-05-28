import socket
import pickle
import struct
import threading
import time
import io
import cv2
import numpy as np
from PIL import Image

class FaceRecognitionClient:
    def __init__(self, server_host, server_port=9999):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.connected = False
        self.frame_buffer = None
        self.recognition_data = None
        self.lock = threading.Lock()
        self.receive_thread = None
        
        # Connection status and error message
        self.connection_error = None
        
        print(f"Face Recognition Client initialized for server {server_host}:{server_port}")
    
    def connect(self):
        """Connect to the face recognition server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_host, self.server_port))
            self.connected = True
            self.connection_error = None
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self.receive_frames)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            print(f"Connected to server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            self.connection_error = str(e)
            print(f"Error connecting to server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
    
    def receive_frames(self):
        """Continuously receive frames from the server"""
        while self.connected and self.client_socket:
            try:
                # Receive message size (4 bytes)
                size_data = self.client_socket.recv(4)
                if not size_data:
                    print("Server closed connection")
                    break
                
                # Unpack size
                size = struct.unpack('>L', size_data)[0]
                
                # Receive data
                data = b''
                while len(data) < size:
                    packet = self.client_socket.recv(min(size - len(data), 4096))
                    if not packet:
                        break
                    data += packet
                
                if len(data) != size:
                    print("Incomplete data received")
                    continue
                
                # Deserialize data
                received_data = pickle.loads(data)
                
                # Update frame buffer and recognition data with thread safety
                with self.lock:
                    self.frame_buffer = received_data.get('frame')
                    self.recognition_data = received_data.get('recognition_data')
                
            except Exception as e:
                print(f"Error receiving data: {e}")
                break
        
        # Connection lost
        self.connected = False
        print("Disconnected from server")
    
    def get_frame(self):
        """Get the latest frame from the server"""
        if not self.connected:
            # Return an error frame if not connected
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Not connected to server", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if self.connection_error:
                cv2.putText(error_img, f"Error: {self.connection_error}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            return buffer.tobytes(), None
        
        # Get frame with thread safety
        with self.lock:
            frame_data = self.frame_buffer
            recognition_data = self.recognition_data
        
        if frame_data is None:
            # Return a waiting frame if no data received yet
            waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_img, "Waiting for video stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', waiting_img)
            return buffer.tobytes(), None
        
        return frame_data, recognition_data
    
    def send_command(self, command):
        """Send a command to the server"""
        if not self.connected or not self.client_socket:
            print("Not connected to server")
            return False
        
        try:
            # Serialize command
            serialized_command = pickle.dumps(command)
            
            # Send size header followed by data
            self.client_socket.sendall(struct.pack('>L', len(serialized_command)) + serialized_command)
            
            # Receive response size
            size_data = self.client_socket.recv(4)
            if not size_data:
                print("No response from server")
                return False
            
            # Unpack size
            size = struct.unpack('>L', size_data)[0]
            
            # Receive response data
            data = b''
            while len(data) < size:
                packet = self.client_socket.recv(min(size - len(data), 4096))
                if not packet:
                    break
                data += packet
            
            if len(data) != size:
                print("Incomplete response received")
                return False
            
            # Deserialize response
            response = pickle.loads(data)
            return response
            
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def add_person(self, person_id, images):
        """Add a new person to the recognition system"""
        # Convert images to bytes if they are PIL Images
        image_bytes = []
        for img in images:
            if isinstance(img, Image.Image):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                image_bytes.append(img_byte_arr.getvalue())
            elif isinstance(img, str):
                # Load image from file path
                with open(img, 'rb') as f:
                    image_bytes.append(f.read())
            else:
                # Assume it's already bytes
                image_bytes.append(img)
        
        # Create command
        command = {
            'type': 'add_person',
            'person_id': person_id,
            'images': image_bytes
        }
        
        # Send command
        return self.send_command(command)
    
    def get_recognition_data(self):
        """Get the latest recognition data"""
        with self.lock:
            return self.recognition_data