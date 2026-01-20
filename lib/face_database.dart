class FaceDatabase {
  static final Map<String, List<double>> _faces = {};

  static void registerFace(String name, List<double> embedding) {
    _faces[name] = embedding;
  }

  static Map<String, List<double>> get faces => _faces;

  static bool get isEmpty => _faces.isEmpty;

  static void clearAllFaces() {
    _faces.clear();
  }
}
