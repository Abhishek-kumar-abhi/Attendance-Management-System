# Face Recognition System Data Flow Diagram

This document illustrates the data flow between different components of the face recognition system, showing how information is processed throughout the application.

## Main Data Flows

### 1. User Registration and Face Enrollment

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  User Input  │────►│  Web App      │────►│  Database      │     │  Recognition   │
│  (Form Data) │     │  (app.py)     │     │  (SQLite)      │     │  Server        │
│              │     │               │     │                │     │                │
└──────────────┘     └───────┬───────┘     └────────────────┘     └────────┬───────┘
                             │                                             │
                             │                                             │
                             ▼                                             ▼
                     ┌───────────────┐                           ┌────────────────┐
                     │               │                           │                │
                     │  Face Images  │─────────────────────────►│  Face Encoding │
                     │  (Uploads)    │                           │  Generation    │
                     │               │                           │                │
                     └───────────────┘                           └────────┬───────┘
                                                                          │
                                                                          │
                                                                          ▼
                                                                 ┌────────────────┐
                                                                 │                │
                                                                 │  Known Faces   │
                                                                 │  Storage       │
                                                                 │                │
                                                                 └────────────────┘
```

### 2. Authentication Flow

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  Login       │────►│  Web App      │────►│  Database      │────►│  Session       │
│  Credentials │     │  (app.py)     │     │  Verification  │     │  Creation      │
│              │     │               │     │                │     │                │
└──────────────┘     └───────────────┘     └────────────────┘     └────────────────┘
                                                                          │
                                                                          │
                                                                          ▼
                                                                 ┌────────────────┐
                                                                 │                │
                                                                 │  User/Admin    │
                                                                 │  Dashboard     │
                                                                 │                │
                                                                 └────────────────┘
```

### 3. Live Recognition Process

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  Camera      │────►│  Recognition  │────►│  Face          │────►│  Feature       │
│  Feed        │     │  Server       │     │  Detection     │     │  Extraction    │
│              │     │               │     │  (MTCNN)       │     │  (FaceNet)     │
└──────────────┘     └───────────────┘     └────────────────┘     └────────┬───────┘
                                                                           │
                                                                           │
                                                                           ▼
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  Web UI      │◄────┤  Recognition  │◄────┤  Identity      │◄────┤  Face          │
│  Display     │     │  Client       │     │  Matching      │     │  Comparison    │
│              │     │               │     │  (SVM)         │     │  with Database │
└──────────────┘     └───────────────┘     └────────────────┘     └────────────────┘
```

### 4. Attendance Recording

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  Recognition │────►│  Web App      │────►│  Attendance    │────►│  Database      │
│  Results     │     │  (app.py)     │     │  Processing    │     │  Storage       │
│              │     │               │     │                │     │                │
└──────────────┘     └───────────────┘     └────────────────┘     └────────────────┘
                                                                          │
                                                                          │
                                                                          ▼
                                                                 ┌────────────────┐
                                                                 │                │
                                                                 │  Attendance    │
                                                                 │  Reports       │
                                                                 │                │
                                                                 └────────────────┘
```

## Data Entities and Their Attributes

### User Data
- Username
- Password (hashed)
- Full Name
- User ID
- Admin Status
- Organization ID

### Organization Data
- Name
- Type
- Creation Date

### Attendance Records
- User ID
- Full Name
- Date
- Time
- Confidence Score
- Organization ID

### Face Recognition Data
- Face Encodings (feature vectors)
- User Identity
- Confidence Scores
- Bounding Box Coordinates

## Configuration Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      config.py                           │
└──────────┬──────────────┬───────────────┬───────────────┘
           │              │               │
           ▼              ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Server Host/ │  │ Database     │  │ Recognition  │
│ Port Settings│  │ Settings     │  │ Parameters   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Client       │  │ Web App      │  │ Face Model   │
│ Connection   │  │ Data Access  │  │ Threshold    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Cross-Component Data Dependencies

1. **Web App → Client → Server**
   - Command data (register face, recognize face)
   - Image data (uploaded photos)
   - Recognition results

2. **Web App → Database**
   - User credentials (securely hashed)
   - Organization information
   - Attendance records
   - Parameterized queries to prevent SQL injection

3. **Server → Recognition Model**
   - Face images for processing
   - Detection and recognition parameters
   - Known face encodings

4. **Recognition Model → Known Faces Storage**
   - Face encodings organized by user ID
   - Feature vectors for comparison

## Security Data Flow

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  User Input  │────►│  Input        │────►│  CSRF          │────►│  Secure        │
│  (Forms/API) │     │  Validation   │     │  Protection    │     │  Processing    │
│              │     │               │     │                │     │                │
└──────────────┘     └───────────────┘     └────────────────┘     └────────────────┘
                                                                          │
                                                                          │
                                                                          ▼
┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│              │     │               │     │                │     │                │
│  Secure      │◄────┤  Parameterized│◄────┤  Environment   │◄────┤  Session       │
│  Response    │     │  Database     │     │  Variables     │     │  Management    │
│              │     │  Queries      │     │                │     │                │
└──────────────┘     └───────────────┘     └────────────────┘     └────────────────┘
```

## Configuration Data Flow (Enhanced)

```
┌──────────────────────────────────────────────────────────┐
│                      config.py                           │
└──────────┬──────────────┬───────────────┬───────────────┘
           │              │               │
           ▼              ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Server Host/ │  │ Database     │  │ Recognition  │
│ Port Settings│  │ Settings     │  │ Parameters   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Client       │  │ Web App      │  │ Face Model   │
│ Connection   │  │ Data Access  │  │ Threshold    │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │
       └────────┬────────┴──────────┬──────┘
                │                   │
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │ Environment  │    │ Security     │
        │ Variables    │    │ Settings     │
        └──────────────┘    └──────────────┘
```

This data flow diagram complements the system architecture and component diagrams by focusing specifically on how data moves through the system, with enhanced security measures implemented throughout.