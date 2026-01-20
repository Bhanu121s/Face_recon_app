import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class FaceStorageService {
  static const String _storageKey = 'registered_faces';

  /// Save a face embedding with name
  static Future<void> saveFace(
    String name,
    List<double> embedding,
  ) async {
    final prefs = await SharedPreferences.getInstance();

    final Map<String, dynamic> faces =
        jsonDecode(prefs.getString(_storageKey) ?? '{}');

    faces[name] = embedding;

    await prefs.setString(_storageKey, jsonEncode(faces));
  }

  /// Get all stored faces
  static Future<Map<String, List<double>>> getAllFaces() async {
    final prefs = await SharedPreferences.getInstance();

    final Map<String, dynamic> faces =
        jsonDecode(prefs.getString(_storageKey) ?? '{}');

    return faces.map(
      (key, value) => MapEntry(
        key,
        List<double>.from(value),
      ),
    );
  }

  /// Delete a single face
  static Future<void> deleteFace(String name) async {
    final prefs = await SharedPreferences.getInstance();

    final Map<String, dynamic> faces =
        jsonDecode(prefs.getString(_storageKey) ?? '{}');

    faces.remove(name);

    await prefs.setString(_storageKey, jsonEncode(faces));
  }

  /// Clear all registered faces
  static Future<void> clearAllFaces() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }
}
