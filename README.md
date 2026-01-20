# Face Recognition App

A real-time face recognition mobile application built with Flutter that detects, recognizes, and stores face data using TensorFlow Lite and Google ML Kit.

## 📱 Features

- **Real-time Face Detection** - Detects faces using Google ML Kit with high accuracy
- **Face Recognition** - Recognizes registered faces using TensorFlow Lite embeddings
- **Multi-face Support** - Detects and recognizes multiple faces simultaneously in a frame
- **Face Registration** - Register new faces with custom names
- **Persistent Storage** - Saves face records using SharedPreferences
- **Camera Switching** - Toggle between front and back cameras
- **Delete All Faces** - Clear all registered face records with confirmation
- **Smooth Performance** - Frame skipping optimization for buttery smooth UX
- **Low Latency** - Processes every 3rd frame for balance between speed and accuracy

## 🎯 Technical Stack

- **Framework**: Flutter 3.10.7+
- **Language**: Dart
- **Face Detection**: Google ML Kit (`google_mlkit_face_detection`)
- **Face Recognition**: TensorFlow Lite (`tflite_flutter`) with 128-dimensional embeddings
- **Camera**: Flutter Camera plugin (`camera`)
- **Storage**: SharedPreferences for persistent data
- **Image Processing**: Dart `image` package

## 📋 Prerequisites

- Flutter 3.10.7 or higher
- Android 21+ or iOS 11+
- Camera permissions enabled

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face_recognition_app
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the app**
   ```bash
   flutter run
   ```

## 📖 Usage

### Register a Face
1. Position your face in front of the camera until you see a green box
2. Tap the **"Register Face"** button
3. Enter your name in the dialog
4. Tap **"Save"** to register
5. Your face will now be recognized in real-time

### Switch Camera
- Tap the **camera flip button** (top-right corner) to switch between front and back cameras

### Delete All Faces
- Tap the **red delete button** (top-left corner)
- Confirm the deletion in the confirmation dialog
- All registered face records will be permanently deleted

### View Recognition
- Recognized faces will display their names above the green bounding boxes
- Unknown faces will show "Unknown"

## 🏗️ Project Structure

```
lib/
├── main.dart                    # App entry point
├── camera_screen.dart           # Main camera UI & face detection logic
├── face_embedding_service.dart  # TFLite model & embedding generation
├── face_matcher.dart            # Face matching using cosine similarity
├── face_database.dart           # In-memory face storage
├── face_storage_service.dart    # Persistent storage with SharedPreferences
└── face_painter.dart            # Custom canvas painter for face boxes

assets/
└── models/
    └── face_embedding.tflite    # TensorFlow Lite embedding model
```

## 🔧 Configuration

### Similarity Threshold
Adjust the face matching threshold in `camera_screen.dart` (line ~107):
```dart
threshold: 0.75,  // Increase for stricter matching, decrease for lenient
```

### Frame Processing Rate
Modify frame skip rate in `camera_screen.dart`:
```dart
static const int _frameSkip = 2; // Process every 3rd frame (0-based)
```

## 🎨 UI Components

| Component | Location | Function |
|-----------|----------|----------|
| Green Bounding Box | Face detection area | Shows detected face boundary |
| Name Label | Above box | Recognized person's name |
| Register Button | Bottom center | Opens registration dialog |
| Flip Camera Button | Top-right | Toggles front/back camera |
| Delete Button | Top-left | Clears all registered faces |

## ⚡ Performance Optimizations

- **Frame Skipping**: Processes every 3rd frame to reduce CPU load by ~67%
- **YUV420 to RGB Conversion**: Efficient image format conversion
- **Lazy Embedding**: Only generates embeddings for detected faces
- **Batch Processing**: Handles multiple faces in parallel

## 🔐 Security & Privacy

- All face embeddings are stored locally on device
- No data is sent to external servers
- SharedPreferences handles encrypted storage on Android/iOS
- Users have full control to delete data anytime

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No face detected" when clicking register | Ensure good lighting and face is clearly visible in the green box |
| App is laggy | Increase frame skip rate or reduce camera resolution |
| Wrong faces being recognized | Increase similarity threshold to 0.8 or higher |
| Camera permission denied | Grant camera permissions in device settings |

## 📊 Model Information

- **Model**: `face_embedding.tflite`
- **Input**: 160×160×3 RGB image
- **Output**: 128-dimensional embedding vector
- **Matching**: Cosine similarity with 0.75 threshold (adjustable)



**App Version**: 1.0.0
