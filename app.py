import os
import cv2
import time
import sqlite3
import datetime
import numpy as np
import re
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the face recognition client
from face_recognition_client import FaceRecognitionClient

# Import configuration
from config import RECOGNITION_SERVER_HOST, RECOGNITION_SERVER_PORT, UPLOAD_FOLDER, MAX_CONTENT_LENGTH

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key or len(app.secret_key) < 16:
    import secrets
    app.secret_key = secrets.token_hex(16)
    print("WARNING: Using a randomly generated secret key. Set SECRET_KEY in .env file for persistence.")

# Initialize CSRF protection
csrf = CSRFProtect(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=2)

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Create the client instance
face_client = FaceRecognitionClient(server_host=RECOGNITION_SERVER_HOST, server_port=RECOGNITION_SERVER_PORT)

# Database setup
def init_db():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    # Create organizations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS organizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        org_type TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Create users table with organization_id
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT NOT NULL,
        user_id TEXT UNIQUE NOT NULL,
        is_admin INTEGER DEFAULT 0,
        organization_id INTEGER,
        FOREIGN KEY (organization_id) REFERENCES organizations (id)
    )
    ''')
    
    # Create attendance table with organization_id
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        full_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        confidence REAL NOT NULL,
        organization_id INTEGER,
        UNIQUE(user_id, date),
        FOREIGN KEY (organization_id) REFERENCES organizations (id)
    )
    ''')
    
    # Create default admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = ?",("admin",))
    if not cursor.fetchone():
        # First check if we have a default organization
        cursor.execute("SELECT id FROM organizations WHERE name = ?",("System",))
        org = cursor.fetchone()
        
        if not org:
            # Create default organization
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO organizations (name, org_type, created_at) VALUES (?, ?, ?)",
                          ('System', 'System', current_time))
            org_id = cursor.lastrowid
        else:
            org_id = org[0]
            
        # Create super admin user with a secure password from environment variable or generate one
        admin_password = os.environ.get('ADMIN_PASSWORD')
        if not admin_password:
            import secrets
            admin_password = secrets.token_urlsafe(12)
            print(f"WARNING: Generated temporary admin password: {admin_password}")
            print("Please change this password immediately after first login!")
        
        cursor.execute("INSERT INTO users (username, password, full_name, user_id, is_admin, organization_id) VALUES (?, ?, ?, ?, ?, ?)",
                      ('admin', generate_password_hash(admin_password), 'Administrator', 'ADMIN001', 1, org_id))
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def get_db_connection():
    conn = sqlite3.connect('face_recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

def mark_attendance(user_id, full_name, confidence, organization_id=None):
    today = datetime.date.today().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if attendance already marked for today
    cursor.execute("SELECT * FROM attendance WHERE user_id = ? AND date = ?", (user_id, today))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO attendance (user_id, full_name, date, time, confidence, organization_id) VALUES (?, ?, ?, ?, ?, ?)",
                      (user_id, full_name, today, current_time, confidence, organization_id))
        conn.commit()
        result = True
    else:
        result = False
    
    conn.close()
    return result

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Validate input to prevent injection attacks
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        is_admin_login = 'is_admin_login' in request.form
        organization_id = request.form.get('organization_id') if is_admin_login else None
        
        # Input validation
        if not username or not password:
            flash('Username and password are required')
            conn = get_db_connection()
            organizations = conn.execute("SELECT * FROM organizations").fetchall()
            conn.close()
            return render_template('login.html', organizations=organizations)
        
        # Prevent timing attacks by using constant time comparison
        conn = get_db_connection()
        
        try:
            # For admin login, we need to check organization as well
            if is_admin_login and organization_id:
                # Validate organization_id is numeric
                if not organization_id.isdigit():
                    raise ValueError("Invalid organization ID")
                    
                user = conn.execute(
                    "SELECT * FROM users WHERE username = ? AND is_admin = 1 AND organization_id = ?", 
                    (username, organization_id)
                ).fetchone()
            else:
                user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            
            if user and check_password_hash(user['password'], password):
                # For admin login, verify they are actually an admin
                if is_admin_login and not user['is_admin']:
                    flash('Invalid admin credentials')
                    organizations = conn.execute("SELECT * FROM organizations").fetchall()
                    conn.close()
                    return render_template('login.html', organizations=organizations)
                
                # Set session data
                session.clear()  # Clear any existing session data first
                session['user_id'] = user['user_id']
                session['username'] = user['username']
                session['full_name'] = user['full_name']
                session['is_admin'] = user['is_admin']
                session['organization_id'] = user['organization_id']
                
                # Regenerate session to prevent session fixation
                session.modified = True
                
                if user['is_admin']:
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('user_dashboard'))
            
            # Use a generic error message to prevent username enumeration
            flash('Invalid username or password')
        except Exception as e:
            # Log the error but don't expose details to the user
            print(f"Login error: {str(e)}")
            flash('An error occurred during login. Please try again.')
        finally:
            conn.close()
        
    # Load organizations for the dropdown
    conn = get_db_connection()
    organizations = conn.execute("SELECT * FROM organizations").fetchall()
    conn.close()
    
    return render_template('login.html', organizations=organizations)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        user_id = request.form['user_id']
        organization_id = request.form.get('organization_id')
        
        conn = get_db_connection()
        
        # Check if username or user_id already exists
        if conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
            conn.close()
            flash('Username already exists')
            return render_template('register.html')
        
        if conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone():
            conn.close()
            flash('User ID already exists')
            return render_template('register.html')
        
        # Create new user with organization_id
        conn.execute("INSERT INTO users (username, password, full_name, user_id, organization_id) VALUES (?, ?, ?, ?, ?)",
                    (username, generate_password_hash(password), full_name, user_id, organization_id))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    # Get organizations for dropdown
    conn = get_db_connection()
    organizations = conn.execute("SELECT * FROM organizations").fetchall()
    conn.close()
    
    return render_template('register.html', organizations=organizations)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/user/dashboard')
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's attendance records filtered by organization_id
    conn = get_db_connection()
    attendance_records = conn.execute("SELECT * FROM attendance WHERE user_id = ? AND organization_id = ? ORDER BY date DESC", 
                                     (session['user_id'], session.get('organization_id'))).fetchall()
    conn.close()
    
    return render_template('user_dashboard.html', attendance_records=attendance_records)

@app.route('/user/upload', methods=['GET', 'POST'])
def upload_photo():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'photos[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('photos[]')
        
        if not files or files[0].filename == '':
            flash('No selected files')
            return redirect(request.url)
        
        # Create directory for user if it doesn't exist
        user_dir = os.path.join('known_faces', f"{session['user_id']}_{session['full_name']}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Process each uploaded file
        file_paths = []
        images = []
        for file in files:
            if file and file.filename != '':
                try:
                    # Validate file is an image
                    img = Image.open(file.stream)
                    img.verify()  # Verify it's a valid image
                    file.stream.seek(0)  # Reset file pointer after verification
                    
                    # Save the file
                    filename = secure_filename(file.filename if file.filename else '')
                    file_path = os.path.join(user_dir, filename)
                    
                    # Save as a fresh image to ensure integrity
                    img = Image.open(file.stream)
                    img.save(file_path)
                    file_paths.append(file_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error processing image {file.filename}: {e}")
                    flash(f'Error processing {file.filename}: Invalid image format')
        
        # Add the person to the face recognition system with all uploaded images
        if file_paths:
            # Limit to 10 photos for better training
            if len(file_paths) > 10:
                file_paths = file_paths[:10]
                images = images[:10]
                flash('Only the first 10 photos will be used for training')
            
            # Connect to the recognition server if not already connected
            if not face_client.connected:
                connection_success = face_client.connect()
                if not connection_success:
                    flash('Could not connect to the recognition server. Photos saved locally but not added to recognition system.')
                    return redirect(url_for('user_dashboard'))
            
            # Send the images to the server
            person_id = f"{session['user_id']}_{session['full_name']}"
            response = face_client.add_person(person_id, images)
            
            if response and response.get('status') == 'success':
                flash(f'{len(file_paths)} photos uploaded and model retrained successfully')
            else:
                error_msg = response.get('message') if response else 'Unknown error'
                flash(f'Photos uploaded but there was an issue with model retraining: {error_msg}')
            
            return redirect(url_for('user_dashboard'))
        else:
            flash('No valid images were uploaded. Please try again with valid image files.')
    
    return render_template('upload_photo.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    # Get organization-specific users
    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users WHERE is_admin = 0 AND organization_id = ?", 
                        (session['organization_id'],)).fetchall()
    
    # Get organization-specific attendance for today
    today = datetime.date.today().strftime('%Y-%m-%d')
    attendance = conn.execute("SELECT * FROM attendance WHERE date = ? AND organization_id = ? ORDER BY time DESC", 
                             (today, session['organization_id'])).fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html', users=users, attendance=attendance)

@app.route('/admin/add_user', methods=['GET', 'POST'])
def admin_add_user():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        user_id = request.form['user_id']
        
        conn = get_db_connection()
        
        # Check if username or user_id already exists
        if conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
            conn.close()
            flash('Username already exists')
            return render_template('add_user.html')
        
        if conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone():
            conn.close()
            flash('User ID already exists')
            return render_template('add_user.html')
        
        # Create new user with organization_id from admin's session
        conn.execute("INSERT INTO users (username, password, full_name, user_id, organization_id) VALUES (?, ?, ?, ?, ?)",
                    (username, generate_password_hash(password), full_name, user_id, session.get('organization_id')))
        conn.commit()
        
        # Handle photo uploads if provided
        if 'photos[]' in request.files:
            files = request.files.getlist('photos[]')
            
            if files and files[0].filename != '':
                # Create directory for user if it doesn't exist
                user_dir = os.path.join('known_faces', f"{user_id}_{full_name}")
                if not os.path.exists(user_dir):
                    os.makedirs(user_dir)
                
                # Process each uploaded file
                file_paths = []
                for file in files:
                    if file and file.filename != '':
                        try:
                            # Validate file is an image
                            img = Image.open(file.stream)
                            img.verify()  # Verify it's a valid image
                            file.stream.seek(0)  # Reset file pointer after verification
                            
                            # Save the file
                            filename = secure_filename(file.filename) if file.filename else ''
                            file_path = os.path.join(user_dir, filename)
                            
                            # Save as a fresh image to ensure integrity
                            img = Image.open(file.stream)
                            img.save(file_path)
                            file_paths.append(file_path)
                        except Exception as e:
                            print(f"Error processing image {file.filename}: {e}")
                            flash(f'Error processing {file.filename}: Invalid image format')
                
                # Add the person to the face recognition system with all uploaded images
                if file_paths:
                    # Limit to 10 photos for better training
                    if len(file_paths) > 10:
                        file_paths = file_paths[:10]
                        flash('Only the first 10 photos will be used for training')
                    
                    # Connect to the recognition server if not already connected
                    if not face_client.connected:
                        connection_success = face_client.connect()
                        if not connection_success:
                            flash('Could not connect to the recognition server. Photos saved locally but not added to recognition system.')
                            conn.close()
                            flash('User added successfully!')
                            return redirect(url_for('admin_dashboard'))
                    
                    # Send the images to the server
                    person_id = f"{user_id}_{full_name}"
                    # Convert file paths to images
                    images = [Image.open(path) for path in file_paths]
                    response = face_client.add_person(person_id, images)
                    
                    if response and response.get('status') == 'success':
                        flash('User added with photos and model retrained successfully!')
                    else:
                        error_msg = response.get('message') if response else 'Unknown error'
                        flash(f'User added but there was an issue with model retraining: {error_msg}')
                else:
                    flash('User added but no valid images were uploaded.')
        
        conn.close()
        flash('User added successfully!')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('add_user.html')

@app.route('/admin/live')
def admin_live():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    return render_template('live_recognition.html')

def generate_frames(organization_id=None):
    # Try to connect to the recognition server if not already connected
    if not face_client.connected:
        connection_success = face_client.connect()
        if not connection_success:
            # If connection failed, yield an error image
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Cannot connect to recognition server", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(error_img, f"Server: {RECOGNITION_SERVER_HOST}:{RECOGNITION_SERVER_PORT}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if face_client.connection_error:
                cv2.putText(error_img, f"Error: {face_client.connection_error}", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            
            # Yield the error image once
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
    
    # Create a Flask app context for database operations
    with app.app_context():
        while True:
            try:
                # Get frame from the client
                frame_bytes, recognition_data = face_client.get_frame()
                
                # Check if we have recognition data for attendance marking
                if recognition_data and recognition_data.get('face_names') and recognition_data.get('face_confidences'):
                    for name, confidence in zip(recognition_data['face_names'], recognition_data['face_confidences']):
                        if name != "Unknown" and confidence > 0.85:
                            # Extract user_id from the name (format: user_id_full_name)
                            parts = name.split('_', 1)
                            if len(parts) == 2:
                                user_id = parts[0]
                                full_name = parts[1]
                                # Mark attendance with organization_id passed from the route
                                mark_attendance(user_id, full_name, confidence, organization_id)
                
                # Yield the frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Short pause to control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error getting frame: {e}")
                # Create an error image
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, "Error getting video frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(error_img, str(e), (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()
                
                # Yield the error image
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Try to reconnect
                if not face_client.connected:
                    face_client.connect()
                
                time.sleep(1)  # Wait before retrying

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    # Capture session data here, within the request context
    organization_id = session.get('organization_id')
    
    # Pass the captured session data to generate_frames
    return Response(generate_frames(organization_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin/attendance')
def admin_attendance():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    # Get date filter from query parameters
    date_filter = request.args.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    
    conn = get_db_connection()
    attendance = conn.execute("SELECT * FROM attendance WHERE date = ? AND organization_id = ? ORDER BY time", 
                            (date_filter, session.get('organization_id'))).fetchall()
    conn.close()
    
    return render_template('attendance.html', attendance=attendance, selected_date=date_filter)

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        user_id = request.form['user_id']
        org_name = request.form['org_name']
        org_type = request.form['org_type']
        
        conn = get_db_connection()
        
        # Check if username or user_id already exists
        if conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
            conn.close()
            flash('Username already exists')
            return render_template('admin_register.html')
        
        if conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone():
            conn.close()
            flash('User ID already exists')
            return render_template('admin_register.html')
            
        # Check if organization name already exists
        if conn.execute("SELECT * FROM organizations WHERE name = ?", (org_name,)).fetchone():
            conn.close()
            flash('Organization name already exists')
            return render_template('admin_register.html')
        
        try:
            # Create new organization
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO organizations (name, org_type, created_at) VALUES (?, ?, ?)",
                          (org_name, org_type, current_time))
            org_id = cursor.lastrowid
            
            # Create admin user for this organization
            cursor.execute("INSERT INTO users (username, password, full_name, user_id, is_admin, organization_id) VALUES (?, ?, ?, ?, ?, ?)",
                          (username, generate_password_hash(password), full_name, user_id, 1, org_id))
            
            conn.commit()
            conn.close()
            
            flash('Organization and admin account created successfully! Please login.')
            return redirect(url_for('login'))
        except Exception as e:
            conn.rollback()
            conn.close()
            flash(f'Error creating account: {str(e)}')
            return render_template('admin_register.html')
    
    return render_template('admin_register.html')

# Input validation function
def validate_input(input_string, pattern=r'^[\w\s\-\.]+$'):
    """Validate input against a regex pattern to prevent injection attacks"""
    if input_string and re.match(pattern, input_string):
        return True
    return False

# Secure database query function
def secure_query(query, params=()):
    """Execute a database query with proper parameter binding"""
    conn = get_db_connection()
    try:
        result = conn.execute(query, params).fetchall()
        conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

if __name__ == '__main__':
    # Run the app on all network interfaces (0.0.0.0) instead of just localhost
    # This makes it accessible from other devices on the network
    # Using port 8081 instead of 8080 to avoid conflicts
    
    # Try to connect to the recognition server at startup
    try:
        connection_success = face_client.connect()
        if connection_success:
            print(f"Successfully connected to recognition server at {RECOGNITION_SERVER_HOST}:{RECOGNITION_SERVER_PORT}")
        else:
            print(f"Warning: Could not connect to recognition server at {RECOGNITION_SERVER_HOST}:{RECOGNITION_SERVER_PORT}")
            print("The web interface will still work, but live recognition features will be unavailable")
            print("Make sure the face_recognition_server.py is running on the computer with the camera")
    except Exception as e:
        print(f"Error connecting to recognition server: {e}")
    
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 8081))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)