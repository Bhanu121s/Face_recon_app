import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FacePainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final Map<int, String> faceNames;

  FacePainter({
    required this.faces,
    required this.imageSize,
    this.faceNames = const {},
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.green;

    final textPaint = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (int i = 0; i < faces.length; i++) {
      final face = faces[i];
      final left =
          size.width - (face.boundingBox.right * size.width / imageSize.width);
      final right =
          size.width - (face.boundingBox.left * size.width / imageSize.width);
      final top = face.boundingBox.top * size.height / imageSize.height;
      final bottom =
          face.boundingBox.bottom * size.height / imageSize.height;
      final rect = Rect.fromLTRB(left, top, right, bottom);

      // Draw face box
      canvas.drawRect(rect, paint);

      // Draw name above box
      final name = faceNames[i] ?? 'Unknown';
      final textSpan = TextSpan(
        text: name,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black87,
        ),
      );

      textPaint.text = textSpan;
      textPaint.layout();

      // Position text above the box
      final textOffset = Offset(left, top - textPaint.height - 5);
      textPaint.paint(canvas, textOffset);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
