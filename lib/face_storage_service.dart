import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class FaceStorageService {
  static const String _storageKey = 'registered_faces';

  /// Save a face embedding with name (allows multiple per person)
  static Future<void> saveFace(
    String name,
    List<double> embedding,
  ) async {
    final prefs = await SharedPreferences.getInstance();

    // Get existing data
    final Map<String, dynamic> faces =
        jsonDecode(prefs.getString(_storageKey) ?? '{}');

    // Initialize list if person doesn't exist
    if (!faces.containsKey(name)) {
      faces[name] = [];
    }

    // Add embedding to person's list
    final embeddings = List.from(faces[name] ?? []);
    embeddings.add(embedding);
    faces[name] = embeddings;

    await prefs.setString(_storageKey, jsonEncode(faces));
    print('✅ Saved embedding for $name');
  }

  /// Get all stored faces
  static Future<Map<String, List<List<double>>>> getAllFaces() async {
    final prefs = await SharedPreferences.getInstance();

    final Map<String, dynamic> faces =
        jsonDecode(prefs.getString(_storageKey) ?? '{}');

    return faces.map(
      (key, value) => MapEntry(
        key,
        (value as List).map((e) => List<double>.from(e)).toList(),
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
