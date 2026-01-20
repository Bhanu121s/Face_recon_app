import 'dart:math';

class FaceMatcher {
  /// Cosine similarity between two vectors
  static double cosineSimilarity(List<double> a, List<double> b) {
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

  /// Returns best match or null
  static String? findBestMatch({
    required List<double> embedding,
    required Map<String, List<double>> database,
    double threshold = 0.5,
  }) {
    String? bestUser;
    double bestScore = -1;

    database.forEach((name, storedEmbedding) {
      final score = cosineSimilarity(embedding, storedEmbedding);
      if (score > bestScore) {
        bestScore = score;
        bestUser = name;
      }
    });

    if (bestScore >= threshold) {
      return bestUser;
    }
    return null;
  }
}
