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
  /// Updated to handle multiple embeddings per person
  static String? findBestMatch({
    required List<double> embedding,
    required Map<String, List<List<double>>> database,
    double threshold = 0.5,
  }) {
    String? bestUser;
    double bestScore = -1;

    // Compare against all stored embeddings
    database.forEach((name, storedEmbeddings) {
      // Find best match for this person across all their embeddings
      for (final storedEmbedding in storedEmbeddings) {
        final score = cosineSimilarity(embedding, storedEmbedding);
        
        if (score > bestScore) {
          bestScore = score;
          bestUser = name;
        }
      }
    });

    if (bestScore >= threshold) {
      print('✅ Best match: $bestUser (score: ${bestScore.toStringAsFixed(2)})');
      return bestUser;
    }
    
    print('⚠️ No match found (best: ${bestScore.toStringAsFixed(2)} < $threshold)');
    return null;
  }
}
