import os
import cv2
import time
import json
import socket
import pickle
import struct
import threading
import numpy as np
import io
from PIL import Image

# Import the face recognition system
from Restnet_face_recognition_model import FaceRecognitionSystem

class FaceRecognitionServer:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.running = False
        
        # Initialize the face recognition system
        self.face_system = FaceRecognitionSystem(known_faces_dir='known_faces', confidence_threshold=0.85)
        
        # Create known_faces directory if it doesn't exist
        if not os.path.exists('known_faces'):
            os.makedirs('known_faces')
            
        print(f"Face Recognition Server initialized on {host}:{port}")
    
    def start(self):
        """Start the server and listen for connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"Server started on {self.host}:{self.port}")
        
        # Start camera thread
        camera_thread = threading.Thread(target=self.camera_stream)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Accept client connections
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Connection from {addr}")
                self.clients.append(client_socket)
                
                # Start a thread to handle client commands
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                print(f"Error accepting connection: {e}")
                if not self.running:
                    break
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients = []
    
    def camera_stream(self):
        """Process camera frames and send to all connected clients"""
        # Open camera with retry mechanism
        max_retries = 3
        retry_count = 0
        cap = None
        
        while retry_count < max_retries and self.running:
            try:
                # Try to open the camera
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    break
                retry_count += 1
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"Error opening camera (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                time.sleep(1)
        
        if not cap or not cap.isOpened():
            # If camera couldn't be opened, send an error image
            print("Error: Could not open camera after multiple attempts")
            # Create a blank image with error message
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera access error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(error_img, "Please check camera connection", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Send error image to all clients
            self.broadcast_frame(error_img)
            return
        
        # Camera opened successfully, process frames
        frame_count = 0
        error_count = 0
        max_errors = 5  # Maximum consecutive errors before giving up
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count > max_errors:
                        print(f"Too many consecutive frame read errors ({error_count})")
                        break
                    continue
                
                # Reset error count on successful frame read
                error_count = 0
                frame_count += 1
                
                # Process frame for face recognition
                processed_frame = self.face_system.process_frame(frame)
                
                # Prepare recognition data
                recognition_data = {
                    'face_names': self.face_system.last_face_names,
                    'face_confidences': [float(conf) for conf in self.face_system.last_face_confidences] if self.face_system.last_face_confidences else []
                }
                
                # Broadcast the frame to all clients
                self.broadcast_frame(processed_frame, recognition_data)
                
                # Limit frame rate to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                error_count += 1
                if error_count > max_errors:
                    break
                time.sleep(0.1)  # Short pause before retrying
        
        # Release the camera when done
        if cap:
            cap.release()
    
    def broadcast_frame(self, frame, recognition_data=None):
        """Send frame to all connected clients"""
        if not self.clients:
            return  # No clients connected
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return
        
        # Prepare data packet
        data = {
            'frame': buffer.tobytes(),
            'recognition_data': recognition_data
        }
        
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        # Prepare message with size header
        message = struct.pack('>L', len(serialized_data)) + serialized_data
        
        # Send to all clients
        disconnected_clients = []
        for client in self.clients:
            try:
                client.sendall(message)
            except:
                # Mark client for removal
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            try:
                client.close()
            except:
                pass
            if client in self.clients:
                self.clients.remove(client)
    
    def handle_client(self, client_socket, addr):
        """Handle commands from a client"""
        while self.running:
            try:
                # Receive command size (4 bytes)
                size_data = client_socket.recv(4)
                if not size_data:
                    break  # Client disconnected
                
                # Unpack size
                size = struct.unpack('>L', size_data)[0]
                
                # Receive command data
                data = b''
                while len(data) < size:
                    packet = client_socket.recv(min(size - len(data), 4096))
                    if not packet:
                        break
                    data += packet
                
                if len(data) != size:
                    print(f"Incomplete data received from {addr}")
                    continue
                
                # Deserialize command
                command = pickle.loads(data)
                
                # Process command
                if command['type'] == 'add_person':
                    self.handle_add_person(command, client_socket)
                elif command['type'] == 'mark_attendance':
                    self.handle_mark_attendance(command, client_socket)
                
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                break
        
        # Remove client from list
        if client_socket in self.clients:
            self.clients.remove(client_socket)
        try:
            client_socket.close()
        except:
            pass
        print(f"Client {addr} disconnected")
    
    def handle_add_person(self, command, client_socket):
        """Handle adding a new person"""
        try:
            person_id = command['person_id']
            images_data = command['images']
            
            # Create directory for user if it doesn't exist
            user_dir = os.path.join('known_faces', person_id)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            # Save images
            file_paths = []
            for i, img_data in enumerate(images_data):
                # Convert bytes to image
                img = Image.open(io.BytesIO(img_data))
                
                # Save image
                file_path = os.path.join(user_dir, f"image_{i}.jpg")
                img.save(file_path)
                file_paths.append(file_path)
            
            # Add person to recognition system
            success = self.face_system.add_new_person(person_id, image_or_path=file_paths)
            
            # Retrain model
            if success:
                self.face_system.retrain_model()
            
            # Send response
            response = {
                'status': 'success' if success else 'error',
                'message': f"Added {person_id} with {len(file_paths)} images" if success else "Failed to add person"
            }
            
            # Serialize and send response
            serialized_response = pickle.dumps(response)
            client_socket.sendall(struct.pack('>L', len(serialized_response)) + serialized_response)
            
        except Exception as e:
            print(f"Error adding person: {e}")
            # Send error response
            response = {
                'status': 'error',
                'message': str(e)
            }
            serialized_response = pickle.dumps(response)
            client_socket.sendall(struct.pack('>L', len(serialized_response)) + serialized_response)
    
    def handle_mark_attendance(self, command, client_socket):
        """Handle marking attendance"""
        # This would be implemented in the web app, not in the server
        pass

if __name__ == "__main__":
    server = FaceRecognitionServer(host='0.0.0.0', port=9999)
    try:
        server.start()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop()
    except Exception as e:
        print(f"Error: {e}")
        server.stop()