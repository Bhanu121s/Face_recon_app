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

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
    _loadSavedFaces();
    _initializeFaceDetector();
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

  Future<void> _loadModel() async {
    await _embeddingService.loadModel();
  }

  Future<void> _loadSavedFaces() async {
    final savedFaces = await FaceStorageService.getAllFaces();
    for (final entry in savedFaces.entries) {
      FaceDatabase.registerFace(entry.key, entry.value);
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
            Map<int, String> newFaceNames = {};

            // Process each face in the frame
            for (int i = 0; i < faces.length; i++) {
              final Face face = faces[i];
              final img.Image cropped = _cropFaceFromRGB(rgbImage, face);
              final List<double> embedding = _embeddingService.getEmbedding(cropped);

              final match = FaceMatcher.findBestMatch(
                embedding: embedding,
                database: FaceDatabase.faces,
                threshold: 0.75,
              );

              newFaceNames[i] = match ?? 'Unknown';

              // Keep track of last embedding for largest face
              if (i == 0) {
                _lastEmbedding = embedding;
                _recognizedName = newFaceNames[i]!;
              }
            }

            debugPrint('✅ Detected ${faces.length} face(s): $newFaceNames');

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
  }

  // ---------------- FACE REGISTRATION (STEP 1) ----------------

  void _showRegisterDialog() {
    if (_lastEmbedding == null) return;

    final TextEditingController controller = TextEditingController();

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Register Face'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'Enter name',
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
                    SnackBar(content: Text('Face registered: $name')),
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

    final x = rect.left.toInt().clamp(0, rgbImage.width - 1);
    final y = rect.top.toInt().clamp(0, rgbImage.height - 1);
    final w = rect.width.toInt().clamp(1, rgbImage.width - x);
    final h = rect.height.toInt().clamp(1, rgbImage.height - y);

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
