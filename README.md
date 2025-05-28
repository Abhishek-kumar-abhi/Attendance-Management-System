# Face Recognition Attendance System

## Overview
This is an advanced Face Recognition Attendance System built with Python, Flask, and deep learning. The system uses FaceNet and MTCNN for accurate face detection and recognition, providing a reliable solution for attendance tracking in educational institutions and organizations.

## Features
- **Robust Face Recognition**: Utilizes FaceNet and MTCNN models for accurate face detection and recognition
- **Multi-User Support**: Supports multiple organizations and users with different access levels
- **Real-time Recognition**: Live face recognition through webcam feed
- **Attendance Tracking**: Automatically records attendance with date, time, and confidence level
- **Admin Dashboard**: Comprehensive dashboard for administrators to manage users and view attendance records
- **User Registration**: Easy registration process with photo upload capability
- **Data Augmentation**: Implements advanced data augmentation techniques for improved recognition accuracy
- **Secure Authentication**: Password-protected login system with role-based access control

## Technology Stack
- **Backend**: Python, Flask
- **Face Recognition**: PyTorch, FaceNet, MTCNN
- **Database**: SQLite
- **Frontend**: HTML, CSS, Bootstrap
- **Computer Vision**: OpenCV, PIL
- **Machine Learning**: SVM classifier for face recognition

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for live recognition)
- CUDA-compatible GPU (optional, for faster processing)

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the web interface at `http://localhost:5000`

### Default Admin Credentials
- Username: admin
- Password: admin123

## Usage

### Admin Functions
1. **Login**: Access the admin dashboard using admin credentials
2. **Register New Users**: Add new users with their details and photos
3. **Manage Organizations**: Create and manage different organizations
4. **View Attendance**: Monitor attendance records for all users

### User Functions
1. **Login**: Access the user dashboard
2. **Upload Photos**: Add multiple photos for better recognition accuracy
3. **View Attendance**: Check personal attendance history

### Live Recognition
1. Navigate to the Live Recognition page
2. The system will automatically detect faces and mark attendance
3. Recognized users will be displayed with their names and confidence scores

## Project Structure
```
├── app.py                         # Main Flask application
├── Restnet_face_recognition_model.py  # Face recognition model implementation
├── requirements.txt               # Python dependencies
├── face_recognition.db            # SQLite database
├── known_faces/                   # Directory for storing user face images
├── static/                        # Static assets
│   └── css/
│       └── styles.css            # CSS styling
└── templates/                     # HTML templates
    ├── add_user.html
    ├── admin_dashboard.html
    ├── admin_register.html
    ├── attendance.html
    ├── live_recognition.html
    ├── login.html
    ├── register.html
    ├── upload_photo.html
    └── user_dashboard.html
```

## Customization

### Confidence Threshold
You can adjust the face recognition confidence threshold in the `app.py` file:
```python
face_system = FaceRecognitionSystem(known_faces_dir='known_faces', confidence_threshold=0.85)
```

### Adding New Features
The modular architecture makes it easy to extend the system with new features:
1. Add new routes in `app.py`
2. Create corresponding HTML templates in the `templates` directory
3. Update the face recognition model in `Restnet_face_recognition_model.py` if needed

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- [FaceNet](https://github.com/timesler/facenet-pytorch) for the face recognition model
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the frontend styling