import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import time
from sklearn.svm import SVC

# NOTE: This file has been modified to force MTCNN face detection to run on CPU
# to avoid CUDA backend issues with torchvision::nms operation.
# The face recognition model still uses GPU if available for better performance.

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir, confidence_threshold=0.85):
        # Set main device for computation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Force CPU for MTCNN to avoid CUDA NMS issues
        # The NMS operation in torchvision is not supported on CUDA in this environment
        self.face_detector_device = torch.device('cpu')
        print(f"Using CPU for face detection to avoid CUDA NMS issues")
        
        # Face detection model - MTCNN with improved parameters
        self.face_detector = MTCNN(
            keep_all=True, 
            device=self.face_detector_device,
            thresholds=[0.6, 0.7, 0.8],
            min_face_size=40,
            post_process=True
        )
        
        # Face recognition model - Use FaceNet instead of ResNet
        self.model = self._create_face_model()
        
        # SVM classifier for face recognition
        self.svm_classifier = None
        
        # Image transformation pipeline with more augmentations
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Additional augmentation transforms for training
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load known faces
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.confidence_threshold = confidence_threshold
        
        # Create known_faces directory if it doesn't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory: {known_faces_dir}")
        
        # Load all faces at initialization
        self._load_all_known_faces()
        
        # Train SVM classifier if we have faces
        if len(self.known_face_encodings) > 0:
            self._train_svm_classifier()
            
        # Store last detection results to reduce flickering
        self.last_face_locations = []
        self.last_face_names = []
        self.last_face_confidences = []
        self.detection_persistence = 5  # Number of frames to persist detections
        self.frame_counter = 0
    
    def _create_face_model(self):
        """Create a FaceNet model for face recognition"""
        # Load pretrained FaceNet model
        model = InceptionResnetV1(pretrained='vggface2').to(self.device)
        model.eval()
        return model
    
    def _train_svm_classifier(self):
        """Train an SVM classifier on known face encodings"""
        if len(self.known_face_encodings) < 2:
            print("Not enough known faces to train SVM classifier")
            return
            
        # Prepare data - ensure all tensors are on CPU for SVM training
        # Stack the encodings and move to CPU if they're not already there
        stacked_encodings = torch.stack(self.known_face_encodings)
        if stacked_encodings.device != torch.device('cpu'):
            X = stacked_encodings.cpu().numpy()
        else:
            X = stacked_encodings.numpy()
            
        y = np.array(self.known_face_names)
        
        # Train SVM with probability support
        self.svm_classifier = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
        self.svm_classifier.fit(X, y)
        print("SVM classifier trained successfully")
        print(f"Trained classifier with {len(self.known_face_names)} people: {', '.join(self.known_face_names)}")
    
    def _load_all_known_faces(self):
        """Load all known faces from the directory structure with data augmentation"""
        print(f"Loading all known faces from {self.known_faces_dir}")
        
        # Process each person's directory
        person_dirs = [d for d in os.listdir(self.known_faces_dir) 
                      if os.path.isdir(os.path.join(self.known_faces_dir, d))]
        
        if not person_dirs:
            print("No person directories found. Please create directories with person names and add face images.")
            print("Example structure:")
            print("known_faces/")
            print("  ├── John/")
            print("  │   ├── john1.jpg")
            print("  │   ├── john2.jpg")
            print("  ├── Alice/")
            print("  │   ├── alice1.jpg")
            print("  │   ├── alice2.jpg")
            return
            
        # Process each person's directory
        for person_name in person_dirs:
            person_dir = os.path.join(self.known_faces_dir, person_name)
            print(f"Processing person: {person_name}")
            encodings = []
            
            # Get image files
            image_files = [f for f in os.listdir(person_dir) 
                          if os.path.isfile(os.path.join(person_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"  No images found for {person_name}")
                continue
                
            # Process each image for this person
            for img_name in image_files:
                img_path = os.path.join(person_dir, img_name)
                try:
                    # Load and process image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Detect faces with confidence scores
                    boxes, probs = self.face_detector.detect(img)
                    if boxes is None or len(boxes) == 0:
                        print(f"  No face detected in {img_path}")
                        continue
                    
                    # Filter faces with high confidence
                    high_conf_indices = [i for i, prob in enumerate(probs) if prob > 0.9]
                    if not high_conf_indices:
                        print(f"  No high confidence faces in {img_path}")
                        continue
                        
                    # Get the largest high-confidence face
                    filtered_boxes = [boxes[i] for i in high_conf_indices]
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in filtered_boxes]
                    largest_face_idx = np.argmax(areas)
                    box = filtered_boxes[largest_face_idx]
                    
                    # Extract face with margin
                    face_width = box[2] - box[0]
                    face_height = box[3] - box[1]
                    # Add 20% margin around the face
                    margin_x = int(face_width * 0.2)
                    margin_y = int(face_height * 0.2)
                    
                    face = img.crop((
                        max(0, box[0] - margin_x),
                        max(0, box[1] - margin_y),
                        min(img.width, box[2] + margin_x),
                        min(img.height, box[3] + margin_y)
                    ))
                    
                    # Create augmented versions of the face
                    # _get_face_encoding will handle device placement internally
                    original_encoding = self._get_face_encoding(face)
                    if original_encoding is not None:
                        encodings.append(original_encoding)
                        
                        # Apply data augmentation to create 5 more samples
                        for i in range(5):
                            # Apply random transformations
                            augmented_face = self._apply_augmentation(face)
                            aug_encoding = self._get_face_encoding(augmented_face)
                            if aug_encoding is not None:
                                encodings.append(aug_encoding)
                        
                        print(f"  Processed {img_name} with augmentations")
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
            
            if encodings:
                # Average the encodings for this person
                avg_encoding = torch.mean(torch.stack(encodings), dim=0)
                # Normalize the averaged encoding
                avg_encoding = nn.functional.normalize(avg_encoding, p=2, dim=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)
                print(f"Added {person_name} with {len(encodings)} face encodings (including augmentations)")
        
        print(f"Successfully loaded {len(self.known_face_names)} known faces")
        if self.known_face_names:
            print(f"Known people: {', '.join(self.known_face_names)}")
    
    def _apply_augmentation(self, face_img):
        """Apply random augmentations to a face image but return PIL Image"""
        # Create a copy of train_transforms without the ToTensor and Normalize steps
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.Resize((160, 160))
        ])
        # Apply only the augmentation transforms, keeping it as PIL Image
        return augmentation_transforms(face_img)
    
    def _get_face_encoding(self, face_img):
        """Extract facial features from a face image (PIL Image or tensor)"""
        # Check if input is already a tensor
        if isinstance(face_img, torch.Tensor):
            # If it's already a tensor, just make sure it has the right shape and normalization
            if face_img.dim() == 3:  # If it's a single image (C,H,W)
                face_tensor = face_img.unsqueeze(0)  # Add batch dimension
            else:
                face_tensor = face_img
        else:
            # Transform PIL Image to tensor
            face_tensor = self.transform(face_img).unsqueeze(0)
    
        # Move to the main device (GPU if available) for the face recognition model
        # This is separate from the face detection which runs on CPU
        face_tensor = face_tensor.to(self.device)
    
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
            # Normalize embedding to unit length for cosine similarity
            embedding = nn.functional.normalize(embedding, p=2, dim=1)[0]
    
        return embedding
    
    def identify_face(self, face_encoding):
        """Identify a face encoding against all known faces using both similarity and SVM"""
        if not self.known_face_encodings:
            return "Unknown", 0.0
            
        # Method 1: SVM classification if available
        if self.svm_classifier is not None:
            # Ensure the tensor is on CPU before converting to numpy
            # This is necessary since SVM operates on CPU
            if face_encoding.device != torch.device('cpu'):
                face_encoding_cpu = face_encoding.cpu()
            else:
                face_encoding_cpu = face_encoding
                
            # Convert tensor to numpy array
            face_np = face_encoding_cpu.numpy().reshape(1, -1)
            
            # Get probabilities from SVM
            proba = self.svm_classifier.predict_proba(face_np)[0]
            max_prob = np.max(proba)
            
            # Get predicted class
            if max_prob > self.confidence_threshold:
                person_idx = np.argmax(proba)
                name = self.svm_classifier.classes_[person_idx]
                return name, max_prob
        
        # Method 2: Cosine similarity (as backup or when SVM is not available)
        similarities = []
        for known_encoding in self.known_face_encodings:
            similarity = torch.dot(face_encoding, known_encoding).item()
            similarities.append(similarity)
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        if confidence >= self.confidence_threshold:
            return self.known_face_names[best_match_idx], confidence
        else:
            return "Unknown", confidence
    
    def process_frame(self, frame):
        """Process a single frame and identify all faces in it"""
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect faces with confidence scores
        # MTCNN's detect method will handle the tensor device internally
        # since we've set the face_detector_device to CPU
        boxes, probs = self.face_detector.detect(pil_image)
        
        # Increment frame counter for tracking detection persistence
        self.frame_counter += 1
        
        # If faces detected in current frame, update stored results
        if boxes is not None and len(boxes) > 0:
            # Reset lists
            self.last_face_locations = []
            self.last_face_names = []
            self.last_face_confidences = []
            
            # Process each detected face with high confidence
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.9:  # Skip faces with low detection confidence
                    continue
                    
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Add margins for better recognition
                face_width = x2 - x1
                face_height = y2 - y1
                margin_x = int(face_width * 0.1)
                margin_y = int(face_height * 0.1)
                
                # Extract face with margin
                face = pil_image.crop((
                    max(0, x1 - margin_x),
                    max(0, y1 - margin_y),
                    min(pil_image.width, x2 + margin_x),
                    min(pil_image.height, y2 + margin_y)
                ))
                
                # Get face encoding
                face_encoding = self._get_face_encoding(face)
                
                if face_encoding is not None:
                    # Identify face
                    name, confidence = self.identify_face(face_encoding)
                    
                    # Store results
                    self.last_face_locations.append((x1, y1, x2, y2))
                    self.last_face_names.append(name)
                    self.last_face_confidences.append(confidence)
        
        # Now draw the boxes using the most recent detection results
        if self.last_face_locations:
            for (x1, y1, x2, y2), name, confidence in zip(
                self.last_face_locations, self.last_face_names, self.last_face_confidences):
                
                # Determine color based on recognition status and confidence
                if name != "Unknown":
                    # Green for known faces, intensity based on confidence
                    green_intensity = min(255, int(confidence * 255))
                    color = (0, green_intensity, 0)
                else:
                    # Red for unknown faces
                    color = (0, 0, 255)
                
                # Draw box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with name and confidence
                confidence_text = f"{confidence * 100:.1f}%"
                label = f"{name}: {confidence_text}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def add_new_person(self, person_name, image_or_path=None, use_webcam=False, num_samples=10):
        """Add a new person to the system with more samples and better quality control"""
        # Create directory for the new person if it doesn't exist
        person_dir = os.path.join(self.known_faces_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"Created directory for {person_name}")
        
        # Method 1: Use provided image(s)
        if image_or_path is not None:
            if isinstance(image_or_path, str):
                # Single image path
                if os.path.isfile(image_or_path):
                    try:
                        # Validate image by opening it
                        img = Image.open(image_or_path)
                        img.verify()  # Verify it's a valid image
                        
                        # Save as a fresh copy to ensure integrity
                        img = Image.open(image_or_path)
                        img_name = os.path.basename(image_or_path)
                        dst_path = os.path.join(person_dir, img_name)
                        img.save(dst_path)
                        print(f"Added image {img_name} for {person_name}")
                    except Exception as e:
                        print(f"Error processing image {image_or_path}: {e}")
                    
            elif isinstance(image_or_path, list):
                # List of image paths
                for img_path in image_or_path:
                    if os.path.isfile(img_path):
                        try:
                            # Validate image by opening it
                            img = Image.open(img_path)
                            img.verify()  # Verify it's a valid image
                            
                            # Save as a fresh copy to ensure integrity
                            img = Image.open(img_path)
                            img_name = os.path.basename(img_path)
                            dst_path = os.path.join(person_dir, img_name)
                            img.save(dst_path)
                            print(f"Added image {img_name} for {person_name}")
                        except Exception as e:
                            print(f"Error processing image {img_path}: {e}")
        
        # Method 2: Capture from webcam with improved quality checks
        if use_webcam:
            print(f"Capturing {num_samples} images for {person_name} from webcam...")
            try:
                # Open webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam")
                    return False
                
                samples_taken = 0
                consecutive_no_face = 0
                while samples_taken < num_samples:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to read frame")
                        break
                    
                    # Convert to PIL for face detection
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Check for faces and their quality
                    boxes, probs = self.face_detector.detect(pil_frame)
                    
                    # Add face detection feedback to frame
                    if boxes is not None and len(boxes) > 0:
                        # Find the largest face with good confidence
                        good_faces = [(box, prob) for box, prob in zip(boxes, probs) if prob > 0.95]
                        
                        if good_faces:
                            # Reset the counter for no faces
                            consecutive_no_face = 0
                            
                            # Get largest face
                            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box, _ in good_faces]
                            largest_idx = np.argmax(areas)
                            box, prob = good_faces[largest_idx]
                            
                            # Draw box around face
                            x1, y1, x2, y2 = [int(b) for b in box]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add quality indicator
                            quality_text = f"Quality: {prob*100:.1f}%"
                            cv2.putText(frame, quality_text, (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Display instructions
                            cv2.putText(frame, "Face detected! Press SPACE to capture", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Show progress
                            cv2.putText(frame, f"Progress: {samples_taken}/{num_samples}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            consecutive_no_face += 1
                            cv2.putText(frame, "Low quality face. Please adjust position/lighting", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        consecutive_no_face += 1
                        cv2.putText(frame, "No face detected. Please look at the camera", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Handle OpenCV GUI issues by trying to show frame and catching errors
                    try:
                        cv2.imshow("Capture Face", frame)
                        key = cv2.waitKey(1) & 0xFF
                    except Exception as e:
                        print(f"GUI error: {e}")
                        # If we can't show the frame, just save samples automatically with a delay
                        time.sleep(1)
                        key = 32  # Space key code
                    
                    if key == 32 and boxes is not None and len(boxes) > 0:  # Space key and face detected
                        # Only save if we have a good quality face
                        good_faces = [(box, prob) for box, prob in zip(boxes, probs) if prob > 0.95]
                        if good_faces:
                            # Save the frame
                            img_path = os.path.join(person_dir, f"{person_name}_{samples_taken+1}.jpg")
                            cv2.imwrite(img_path, frame)
                            print(f"Saved image {samples_taken+1} for {person_name}")
                            samples_taken += 1
                            # Longer delay to allow changing pose/expression
                            time.sleep(1.0)
                    
                    elif key == 27:  # Esc key
                        print("Capture canceled")
                        break
                
                # Release resources
                cap.release()
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                
            except Exception as e:
                print(f"Error during webcam capture: {e}")
                return False
        
        # Reload all known faces and train classifier
        print("Retraining face recognition model with new images...")
        self._load_all_known_faces()
        if len(self.known_face_encodings) >= 2:
            self._train_svm_classifier()
            print("Model successfully retrained with new face data")
            return True
        else:
            print("Not enough face data to train the classifier (need at least 2 people)")
            return False
    
    def retrain_model(self):
        """Explicitly retrain the model with all known faces"""
        print("Retraining face recognition model...")
        self._load_all_known_faces()
        if len(self.known_face_encodings) >= 2:
            self._train_svm_classifier()
            print("Model successfully retrained")
            return True
        else:
            print("Not enough face data to train the classifier (need at least 2 people)")
            return False
    
    def run_live_recognition(self, camera_id=0, write_output=False):
        """Run continuous face recognition on live camera feed"""
        # Check if we have any known faces
        if not self.known_face_encodings:
            print("Warning: No known faces loaded. The system will only detect faces but won't recognize anyone.")
        
        # Open video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            print("Let's try using the system without GUI display to capture still images instead.")
            self._alternative_processing()
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened: {width}x{height} at {fps} FPS")
        print("Press 'q' to quit, 's' to save a still image")
        
        # Output video writer
        writer = None
        if write_output:
            output_path = f"face_recognition_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            print(f"Writing output to {output_path}")
        
        # Performance metrics
        frame_count = 0
        start_time = time.time()
        
        # Process every nth frame to maintain performance but reduce flickering
        process_frequency = 3  # Process every 3rd frame (adjust based on your system's capabilities)
        frame_idx = 0
        
        # Main processing loop
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Process only every nth frame to maintain performance
                if frame_idx % process_frequency == 0:
                    # Process current frame and update face tracking data
                    processed_frame = self.process_frame(frame)
                else:
                    # Use the last detection results to annotate the frame without reprocessing
                    processed_frame = frame.copy()
                    if self.last_face_locations:
                        for (x1, y1, x2, y2), name, confidence in zip(
                            self.last_face_locations, self.last_face_names, self.last_face_confidences):
                            
                            # Determine color based on recognition status and confidence
                            if name != "Unknown":
                                green_intensity = min(255, int(confidence * 255))
                                color = (0, green_intensity, 0)
                            else:
                                color = (0, 0, 255)
                            
                            # Draw box
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Label
                            confidence_text = f"{confidence * 100:.1f}%"
                            label = f"{name}: {confidence_text}"
                            
                            # Draw label background
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(processed_frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, cv2.FILLED)
                            
                            # Draw label text
                            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Increment frame index
                frame_idx += 1
                
                # Calculate and display FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(processed_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Reset counters
                    frame_count = 0
                    start_time = time.time()
                
                # Display info about known people
                info_text = f"Tracking {len(self.known_face_names)} people"
                cv2.putText(processed_frame, info_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame if requested
                if writer:
                    writer.write(processed_frame)
                
                # Try to display frame, but handle exceptions for headless systems
                try:
                    cv2.imshow("Face Recognition System", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as e:
                    print(f"GUI error: {e}")
                    print("Switching to non-GUI mode...")
                    break
                
                # Handle key presses
                if key == ord('q'):
                    print("Quitting")
                    break
                elif key == ord('s'):
                    # Save a still image
                    img_path = f"face_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(img_path, processed_frame)
                    print(f"Saved frame to {img_path}")
        
        except Exception as e:
            print(f"Error during live recognition: {e}")
        
        finally:
            # Release resources
            cap.release()
            if writer:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def _alternative_processing(self):
        """Alternative processing mode when GUI is not available"""
        print("\nRunning in non-GUI mode")
        print("1. Add a new person")
        print("2. Process a single image")
        print("3. Continuous processing (save images but don't display)")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            name = input("Enter person's name: ")
            self.add_new_person(name, use_webcam=True, num_samples=10)
            self._alternative_processing()
            
        elif choice == "2":
            image_path = input("Enter image path: ")
            if os.path.isfile(image_path):
                try:
                    frame = cv2.imread(image_path)
                    processed_frame = self.process_frame(frame)
                    output_path = f"processed_{os.path.basename(image_path)}"
                    cv2.imwrite(output_path, processed_frame)
                    print(f"Processed image saved to {output_path}")
                except Exception as e:
                    print(f"Error processing image: {e}")
            else:
                print(f"File not found: {image_path}")
            self._alternative_processing()
            
        elif choice == "3":
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open camera")
                    self._alternative_processing()
                    return
                
                print("Capturing frames. Press Ctrl+C to stop.")
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error reading frame")
                        break
                    
                    # Process every 10th frame to avoid too many files
                    if frame_count % 10 == 0:
                        processed_frame = self.process_frame(frame)
                        output_path = f"frame_{frame_count}.jpg"
                        cv2.imwrite(output_path, processed_frame)
                        print(f"Saved {output_path}")
                    
                    frame_count += 1
                    time.sleep(0.1)  # Slow down capture
                    
            except KeyboardInterrupt:
                print("\nContinuous capture stopped.")
            except Exception as e:
                print(f"Error during continuous capture: {e}")
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            self._alternative_processing()
            
        elif choice == "4":
            print("Exiting.")
            return
        else:
            print("Invalid choice.")
            self._alternative_processing()


def main():
    # Directory to store known faces
    known_faces_dir = "known_faces"
    
    # Create and initialize the face recognition system
    face_system = FaceRecognitionSystem(known_faces_dir=known_faces_dir, confidence_threshold=0.85)
    
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--add', help='Add a new person using webcam', action='store_true')
    parser.add_argument('--name', help='Name of the person to add')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to take for new person')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--record', action='store_true', help='Record output video')
    parser.add_argument('--image', help='Path to an image file to process')
    args = parser.parse_args()
    
    if args.add and args.name:
        # Add a new person
        face_system.add_new_person(args.name, use_webcam=True, num_samples=args.samples)
    elif args.image:
        # Process a single image
        if os.path.isfile(args.image):
            try:
                print(f"Processing image: {args.image}")
                frame = cv2.imread(args.image)
                processed_frame = face_system.process_frame(frame)
                output_path = f"processed_{os.path.basename(args.image)}"
                cv2.imwrite(output_path, processed_frame)
                print(f"Processed image saved to {output_path}")
                
                # Try to display the result
                try:
                    cv2.imshow("Processed Image", processed_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print("Could not display image (running in headless mode)")
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print(f"File not found: {args.image}")
    else:
        # Run live recognition
        face_system.run_live_recognition(camera_id=args.camera, write_output=args.record)


if __name__ == "__main__":
    main()