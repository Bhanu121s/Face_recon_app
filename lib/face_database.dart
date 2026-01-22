import 'dart:math';

class FaceDatabase {
  static final Map<String, List<List<double>>> _faces = {};

  // Register a face (allows multiple embeddings per person)
  static void registerFace(String name, List<double> embedding) {
    if (!_faces.containsKey(name)) {
      _faces[name] = [];
    }
    
    // Add embedding if not already similar to existing ones
    final isDuplicate = _faces[name]!.any((existing) {
      return _cosineSimilarity(existing, embedding) > 0.85;
    });
    
    if (!isDuplicate) {
      _faces[name]!.add(embedding);
      print('✅ Added embedding for $name (total: ${_faces[name]!.length})');
    } else {
      print('⚠️ Similar embedding for $name already exists');
    }
  }

  static Map<String, List<List<double>>> get faces => _faces;

  static bool get isEmpty => _faces.isEmpty;

  static void clearAllFaces() {
    _faces.clear();
  }

  // Helper to calculate cosine similarity
  static double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dot / (sqrt(normA) * sqrt(normB));
  }
}
