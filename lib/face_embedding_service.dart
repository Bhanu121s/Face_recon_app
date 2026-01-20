import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class FaceEmbeddingService {
  static const int inputSize = 160;
  static const int embeddingSize = 128;

  Interpreter? _interpreter;

  /// Load TFLite model
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/face_embedding.tflite',
      );
      print('✅ TFLite model loaded');
    } catch (e) {
      print('❌ Failed to load model: $e');
      rethrow;
    }
  }

  bool get isLoaded => _interpreter != null;

  // ============================================================
  // MAIN ENTRY — used by CameraScreen
  // ============================================================
  Future<List<double>> getEmbeddingFromCameraImage(
    CameraImage cameraImage,
    Face face,
  ) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not loaded');
    }

    final img.Image rgbImage = _convertCameraImage(cameraImage);

    final Rect box = face.boundingBox;

    final img.Image cropped = img.copyCrop(
      rgbImage,
      x: box.left.toInt().clamp(0, rgbImage.width - 1),
      y: box.top.toInt().clamp(0, rgbImage.height - 1),
      width: box.width.toInt().clamp(1, rgbImage.width),
      height: box.height.toInt().clamp(1, rgbImage.height),
    );

    return getEmbedding(cropped);
  }

  // ============================================================
  // CORE EMBEDDING FUNCTION
  // ============================================================
  List<double> getEmbedding(img.Image faceImage) {
    try {
      final img.Image resized =
          img.copyResize(faceImage, width: inputSize, height: inputSize);

      final Float32List input =
          Float32List(1 * inputSize * inputSize * 3);

      int index = 0;
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);

          input[index++] = (pixel.r - 127.5) / 128.0;
          input[index++] = (pixel.g - 127.5) / 128.0;
          input[index++] = (pixel.b - 127.5) / 128.0;
        }
      }

      final output =
          List.filled(embeddingSize, 0.0).reshape([1, embeddingSize]);

      _interpreter!.run(
        input.reshape([1, inputSize, inputSize, 3]),
        output,
      );

      return List<double>.from(output[0]);
    } catch (e) {
      print('❌ Error generating embedding: $e');
      rethrow;
    }
  }

  // ============================================================
  // CAMERA IMAGE → RGB IMAGE (Y plane only – fast & stable)
  // ============================================================
  img.Image _convertCameraImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final img.Image imgImage =
        img.Image(width: width, height: height);

    final Uint8List bytes = image.planes[0].bytes;
    int index = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixel = bytes[index++];
        imgImage.setPixelRgb(x, y, pixel, pixel, pixel);
      }
    }

    return imgImage;
  }
}
