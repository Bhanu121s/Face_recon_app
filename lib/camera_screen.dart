import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'main.dart';
import 'face_embedding_service.dart';
import 'face_painter.dart';
import 'face_matcher.dart';
import 'face_database.dart';
import 'face_storage_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;

  bool _isCameraInitialized = false;
  bool _isProcessingFrame = false;
  bool _isFrontCamera = true;
  bool _isCameraSwitching = false;
  int _frameCount = 0;
  static const int _frameSkip = 2; // Process every 3rd frame

  late FaceDetector _faceDetector;
  final FaceEmbeddingService _embeddingService = FaceEmbeddingService();

  List<Face> _faces = [];
  List<double>? _lastEmbedding;
  String _recognizedName = 'Unknown';
  Map<int, String> _faceNames = {}; // Maps face index to recognized name
  Map<int, List<String>> _recentMatches = {}; // Stores recent matches for smoothing
  static const int _smoothingFrames = 3; // Number of frames to smooth over
  static const double _minFaceSize = 100.0; // Minimum face width in pixels

  @override
  void initState() {
    super.initState();
    _initializeFaceDetector();
    _initializeAppWithData();
  }

  Future<void> _initializeAppWithData() async {
    debugPrint('🚀 Initializing app...');
    // Load model first
    await _loadModel();
    debugPrint('✅ Model loaded');
    
    // Load saved faces before camera starts
    await _loadSavedFaces();
    debugPrint('✅ Saved faces loaded: ${FaceDatabase.faces.length} people');
    
    // Finally, start camera
    if (mounted) {
      await _initializeCamera();
      debugPrint('✅ Camera initialized');
    }
  }

  void _initializeFaceDetector() {
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableLandmarks: false,
        enableContours: false,
        enableClassification: false,
        minFaceSize: 0.1,
      ),
    );
  }

  // Check if face is large enough and good quality
  bool _isFaceQuality(Face face, Size imageSize) {
    final faceWidth = face.boundingBox.width;
    final minWidth = _minFaceSize;
    
    // Face must be large enough
    if (faceWidth < minWidth) {
      debugPrint('⚠️ Face too small: ${faceWidth.toStringAsFixed(1)}px (min: $minWidth)');
      return false;
    }
    
    return true;
  }

  // Get smoothed recognition result from recent matches
  String _getSmoothedResult(int faceIndex, String currentMatch) {
    if (!_recentMatches.containsKey(faceIndex)) {
      _recentMatches[faceIndex] = [];
    }
    
    final matches = _recentMatches[faceIndex]!;
    matches.add(currentMatch);
    
    // Keep only recent matches
    if (matches.length > _smoothingFrames) {
      matches.removeAt(0);
    }
    
    // Return most common match
    final counts = <String, int>{};
    for (final match in matches) {
      counts[match] = (counts[match] ?? 0) + 1;
    }
    
    return counts.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  Future<void> _loadModel() async {
    await _embeddingService.loadModel();
  }

  Future<void> _loadSavedFaces() async {
    try {
      final savedFaces = await FaceStorageService.getAllFaces();
      debugPrint('📁 Retrieved ${savedFaces.length} people from storage');
      
      int totalEmbeddings = 0;
      for (final entry in savedFaces.entries) {
        debugPrint('   Loading "${entry.key}": ${entry.value.length} embeddings');
        for (final embedding in entry.value) {
          FaceDatabase.registerFace(entry.key, embedding);
          totalEmbeddings++;
        }
      }
      
      debugPrint('✅ Loaded $totalEmbeddings total embeddings into FaceDatabase');
      debugPrint('   FaceDatabase now contains: ${FaceDatabase.faces}');
    } catch (e, stackTrace) {
      debugPrint('❌ Error loading faces: $e');
      debugPrint('Stack trace: $stackTrace');
    }
  }

  Future<void> _initializeCamera() async {
    final lensDirection = _isFrontCamera 
        ? CameraLensDirection.front 
        : CameraLensDirection.back;
    
    final camera = cameras.firstWhere(
      (cam) => cam.lensDirection == lensDirection,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _controller.initialize();

    _controller.startImageStream((CameraImage image) async {
      _frameCount++;
      
      // Skip frames for better performance
      if (_frameCount % (_frameSkip + 1) != 0) return;
      
      if (_isProcessingFrame) return;
      _isProcessingFrame = true;

      try {
        final inputImage = _inputImageFromCameraImage(image);

        final faces = await _faceDetector.processImage(inputImage);

        if (mounted) {
          setState(() {
            _faces = faces;
          });
        }

        // ---------- FACE EMBEDDING + MATCHING ----------
        if (faces.isNotEmpty && _embeddingService.isLoaded) {
          try {
            final img.Image rgbImage = _convertCameraImageToRGB(image);
            final Size imageSize = Size(rgbImage.width.toDouble(), rgbImage.height.toDouble());
            Map<int, String> newFaceNames = {};

            // Only process largest face for more stable recognition
            if (faces.isNotEmpty) {
              final Face largestFace = _getLargestFace(faces);
              
              // Check face quality
              if (_isFaceQuality(largestFace, imageSize)) {
                final img.Image cropped = _cropFaceFromRGB(rgbImage, largestFace);
                final List<double>? embedding = _embeddingService.getEmbedding(cropped);

                if (embedding != null && embedding.isNotEmpty) {
                  _lastEmbedding = embedding;

                  final match = FaceMatcher.findBestMatch(
                    embedding: embedding,
                    database: FaceDatabase.faces,
                    threshold: 0.72,
                  );

                  // Apply smoothing for stable results
                  final smoothedResult = _getSmoothedResult(0, match ?? 'Unknown');
                  newFaceNames[0] = smoothedResult;
                  _recognizedName = smoothedResult;

                  debugPrint('✅ Match: $smoothedResult');
                } else {
                  debugPrint('❌ Failed to generate embedding');
                  newFaceNames[0] = 'Unknown';
                }
              } else {
                newFaceNames[0] = 'Unknown';
              }
            }

            if (mounted) {
              setState(() {
                _faceNames = newFaceNames;
              });
            }
          } catch (e) {
            debugPrint('❌ Embedding error: $e');
          }
        }
      } catch (e) {
        debugPrint('Frame processing error: $e');
      } finally {
        _isProcessingFrame = false;
      }
    });

    setState(() {
      _isCameraInitialized = true;
    });

    // Log initialization completion
    debugPrint('🎥 Camera stream started and ready for face detection');
  }

  void _showRegisterDialog() {
    if (_lastEmbedding == null) return;

    final TextEditingController controller = TextEditingController();

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Register Face'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: controller,
                decoration: const InputDecoration(
                  hintText: 'Enter name',
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                '💡 Tip: For best accuracy, register the same face from:\n'
                '• Different lighting conditions\n'
                '• Different angles (front, left, right)\n'
                '• Different distances from camera\n\n'
                'This helps the system recognize you reliably.',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final name = controller.text.trim();
              if (name.isNotEmpty) {
                // Save to persistent storage
                await FaceStorageService.saveFace(name, _lastEmbedding!);
                
                // Update in-memory database
                FaceDatabase.registerFace(name, _lastEmbedding!);
                
                if (mounted) {
                  setState(() {
                    _recognizedName = name;
                  });
                }
                
                if (mounted) {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('Face registered: $name\n✅ Register from different angles for better accuracy'),
                      duration: const Duration(seconds: 3),
                    ),
                  );
                }
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  // ---------------- HELPER METHODS ----------------

  InputImage _inputImageFromCameraImage(CameraImage image) {
    final allBytes = <int>[];
    for (final plane in image.planes) {
      allBytes.addAll(plane.bytes);
    }

    final bytes = Uint8List.fromList(allBytes);

    final imageSize =
        Size(image.width.toDouble(), image.height.toDouble());

    final rotation =
        InputImageRotationValue.fromRawValue(
              _controller.description.sensorOrientation,
            ) ??
            InputImageRotation.rotation0deg;

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: imageSize,
        rotation: rotation,
        format: InputImageFormat.nv21,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );
  }

  Face _getLargestFace(List<Face> faces) {
    Face largest = faces.first;
    double maxArea = 0;

    for (final face in faces) {
      final area =
          face.boundingBox.width * face.boundingBox.height;
      if (area > maxArea) {
        maxArea = area;
        largest = face;
      }
    }
    return largest;
  }

  img.Image _convertCameraImageToRGB(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final img.Image imgImage = img.Image(width: width, height: height);

    final Uint8List yPlane = cameraImage.planes[0].bytes;
    int index = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixel = yPlane[index++];
        imgImage.setPixelRgb(x, y, pixel, pixel, pixel);
      }
    }

    return imgImage;
  }

  img.Image _cropFaceFromRGB(img.Image rgbImage, Face face) {
    final rect = face.boundingBox;
    
    // Add 20% padding around face for better feature capture
    final padding = 0.2;
    final paddedWidth = rect.width * padding;
    final paddedHeight = rect.height * padding;

    final x = (rect.left - paddedWidth).toInt().clamp(0, rgbImage.width - 1);
    final y = (rect.top - paddedHeight).toInt().clamp(0, rgbImage.height - 1);
    final w = (rect.width * (1 + padding * 2)).toInt().clamp(1, rgbImage.width - x);
    final h = (rect.height * (1 + padding * 2)).toInt().clamp(1, rgbImage.height - y);

    return img.copyCrop(
      rgbImage,
      x: x,
      y: y,
      width: w,
      height: h,
    );
  }

  @deprecated
  img.Image _cropFace(CameraImage image, Face face) {
    final img.Image fullImage = img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes[0].bytes.buffer,
      format: img.Format.uint8,
    );

    final rect = face.boundingBox;

    final x = rect.left.toInt().clamp(0, image.width - 1);
    final y = rect.top.toInt().clamp(0, image.height - 1);
    final w = rect.width.toInt().clamp(0, image.width - x);
    final h = rect.height.toInt().clamp(0, image.height - y);

    return img.copyCrop(
      fullImage,
      x: x,
      y: y,
      width: w,
      height: h,
    );
  }

  Future<void> _flipCamera() async {
    setState(() {
      _isCameraSwitching = true;
    });
    
    try {
      _isFrontCamera = !_isFrontCamera;
      await _controller.stopImageStream();
      await _controller.dispose();
      await Future.delayed(const Duration(milliseconds: 300)); // Brief delay
      await _initializeCamera();
    } finally {
      if (mounted) {
        setState(() {
          _isCameraSwitching = false;
        });
      }
    }
  }

  void _deleteAllFaces() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete All Faces?'),
        content: const Text(
          'This will permanently delete all registered face records. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              // Clear from persistent storage
              await FaceStorageService.clearAllFaces();
              
              // Clear in-memory database
              FaceDatabase.clearAllFaces();
              
              if (mounted) {
                setState(() {
                  _recognizedName = 'Unknown';
                });
              }
              
              if (mounted) {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('All faces deleted')),
                );
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    _faceDetector.close();
    super.dispose();
  }

  // ---------------- UI ----------------

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final previewSize = _controller.value.previewSize!;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_controller),

          CustomPaint(
            painter: FacePainter(
              faces: _faces,
              imageSize: Size(
                previewSize.height,
                previewSize.width,
              ),
              faceNames: _faceNames,
            ),
          ),

          // Register button
          Positioned(
            bottom: 120,
            left: 0,
            right: 0,
            child: Center(
              child: ElevatedButton(
                onPressed: () {
                  if (_lastEmbedding == null) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Please position your face in front of the camera'),
                      ),
                    );
                  } else {
                    _showRegisterDialog();
                  }
                },
                child: Text(
                  _lastEmbedding == null ? 'Register Face (No face detected)' : 'Register Face',
                ),
              ),
            ),
          ),

          // Flip Camera button
          Positioned(
            top: 20,
            right: 20,
            child: FloatingActionButton(
              heroTag: 'flip_camera',
              mini: true,
              onPressed: _flipCamera,
              child: const Icon(Icons.flip_camera_android),
            ),
          ),

          // Delete Faces button
          Positioned(
            top: 20,
            left: 20,
            child: FloatingActionButton(
              heroTag: 'delete_faces',
              mini: true,
              onPressed: _deleteAllFaces,
              backgroundColor: Colors.red,
              child: const Icon(Icons.delete),
            ),
          ),

          // Loading overlay during camera switch
          if (_isCameraSwitching)
            Container(
              color: Colors.black.withOpacity(0.7),
              child: const Center(
                child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
