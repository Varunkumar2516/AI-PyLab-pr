// homepage.dart
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

// Import your login page (adjust the path if needed)
import 'login.dart';

class Homepage extends StatefulWidget {
  static const routeName = '/home';

  const Homepage({Key? key}) : super(key: key);

  @override
  State<Homepage> createState() => _HomepageState();
}

class _HomepageState extends State<Homepage> {
  final ImagePicker _picker = ImagePicker();

  // UI state
  bool _loading = false;
  String? _selectedFileName;
  String? _sitrep;
  String? _errorMessage;

  Timer? _pollTimer;

  // Base URL: for Android emulator use 10.0.2.2, for web use localhost.
  // Update this to match your server address if required.
  // final String _baseUrl = kIsWeb ? 'http://localhost:5000' : 'http://10.0.2.2:5000';
  final String _baseUrl = 'http://202.164.52.253:5000';

  @override
  void initState() {
    super.initState();
    _fetchSitrep(); // fetch once on load
    _pollTimer = Timer.periodic(const Duration(minutes: 1), (_) => _fetchSitrep());
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  // Opens the device camera to record a video (image_picker handles record UI)
  Future<void> _openCamera() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.camera,
        maxDuration: const Duration(minutes: 5),
      );
      if (video == null) {
        _showSnack('No video recorded.');
        return;
      }
      setState(() {
        _selectedFileName = video.name;
        _errorMessage = null;
      });
      await _uploadFile(File(video.path), video.name);
    } catch (e) {
      _showSnack('Failed to record video: $e');
    }
  }

  // Opens a file picker to select a video file from storage
  Future<void> _uploadFootage() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );
      if (result == null || result.files.isEmpty) {
        _showSnack('No video selected.');
        return;
      }
      final path = result.files.single.path;
      if (path == null) {
        _showSnack('Selected file path is null.');
        return;
      }
      final name = result.files.single.name;
      setState(() {
        _selectedFileName = name;
        _errorMessage = null;
      });
      await _uploadFile(File(path), name);
    } catch (e) {
      _showSnack('Failed to pick video: $e');
    }
  }

  // Upload a file to the server with retries, replicating the HTML logic
  Future<void> _uploadFile(File file, String filename, {int retries = 3}) async {
    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    final uri = Uri.parse('$_baseUrl/upload');

    for (int attempt = 0; attempt < retries; attempt++) {
      try {
        final request = http.MultipartRequest('POST', uri);
        // 'video' is the same form key used in the HTML
        final multipartFile = await http.MultipartFile.fromPath('video', file.path);
        request.files.add(multipartFile);

        final streamedResponse = await request.send();
        final responseBody = await streamedResponse.stream.bytesToString();

        if (streamedResponse.statusCode >= 200 && streamedResponse.statusCode < 300) {
          setState(() {
            _sitrep = responseBody;
            _errorMessage = null;
          });
          _showSnack('Upload successful.');
          return;
        } else {
          // treat as error to retry
          throw HttpException('HTTP ${streamedResponse.statusCode}: $responseBody');
        }
      } catch (error) {
        // Last attempt -> surface error
        if (attempt == retries - 1) {
          setState(() {
            _errorMessage = 'Error uploading video: ${error.toString()}';
          });
        } else {
          // exponential backoff-like delay
          await Future.delayed(Duration(milliseconds: 1000 * (attempt + 1)));
        }
      }
    }

    setState(() {
      _loading = false;
    });
  }

  // Fetch SITREP from server with retries (like fetchWithRetry in HTML)
  Future<void> _fetchSitrep({int retries = 3}) async {
    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    final uri = Uri.parse('$_baseUrl/get_sitrep');

    for (int attempt = 0; attempt < retries; attempt++) {
      try {
        final response = await http.get(uri).timeout(const Duration(seconds: 10));
        if (response.statusCode >= 200 && response.statusCode < 300) {
          setState(() {
            _sitrep = response.body.isNotEmpty ? response.body : 'No SITREP available yet.';
            _errorMessage = null;
          });
          break;
        } else {
          throw HttpException('HTTP ${response.statusCode}: ${response.reasonPhrase}');
        }
      } catch (error) {
        if (attempt == retries - 1) {
          setState(() {
            _errorMessage =
                'Error fetching SITREP: ${error.toString()}. The API might be overloaded or unavailable.';
            _sitrep = null;
          });
        } else {
          await Future.delayed(Duration(milliseconds: 1000 * (attempt + 1)));
        }
      }
    }

    setState(() {
      _loading = false;
    });
  }

  // Sign out action: pop to first route then replace it with LoginPage
  Future<void> _signOut() async {
    try {
      // If you're using Firebase Auth, uncomment and use:
      // await FirebaseAuth.instance.signOut();
    } catch (e) {
      // ignore sign out error — we'll still navigate to login
    }

    if (!mounted) return;

    // Pop until the first route on the stack
    try {
      Navigator.of(context).popUntil((route) => route.isFirst);
    } catch (e) {
      // ignore any popUntil errors
    }

    // Replace that first route with LoginPage so user can't navigate back
    try {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => Login()),
      );
    } catch (e) {
      // As a last resort, attempt to clear the stack and push LoginPage via pushAndRemoveUntil
      try {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (_) => Login()),
          (Route<dynamic> route) => false,
        );
      } catch (_) {
        _showSnack('Could not navigate to login. Please restart the app.');
      }
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  // Styling constants to match the screenshot
  final Color _bg = const Color(0xFFF7EEF6); // pale pink-ish
  final Color _buttonBg = const Color(0xFFF2EBF5); // light lavender pill
  final Color _buttonText = const Color(0xFF5B3E82); // purple text
  final double _pillWidth = 220;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: SafeArea(
        child: Stack(
          children: [
            // Top title "Home Page" aligned left like screenshot
            Padding(
              padding: const EdgeInsets.only(left: 16.0, top: 8.0),
              child: Text(
                'Home Page',
                style: TextStyle(
                  color: Colors.black87,
                  fontSize: 24,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),

            // Center buttons + status
            Center(
              child: SingleChildScrollView(
                physics: const NeverScrollableScrollPhysics(),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildPillButton(
                      label: 'Open Camera',
                      onPressed: _openCamera,
                    ),
                    const SizedBox(height: 18),
                    _buildPillButton(
                      label: 'Upload Footage',
                      onPressed: _uploadFootage,
                    ),
                    const SizedBox(height: 20),
                    // Selected file name
                    if (_selectedFileName != null)
                      Text(
                        'Selected file: $_selectedFileName',
                        style: const TextStyle(fontSize: 14),
                      ),
                    const SizedBox(height: 12),
                    // Loading indicator
                    if (_loading) const CircularProgressIndicator(),
                    const SizedBox(height: 12),
                    // SITREP display
                    Container(
                      width: MediaQuery.of(context).size.width * 0.78,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(8),
                        boxShadow: [
                          BoxShadow(color: Colors.black12, blurRadius: 6, offset: Offset(0, 2)),
                        ],
                      ),
                      child: _sitrep != null
                          ? SingleChildScrollView(
                              child: Text(
                                _sitrep!,
                                style: const TextStyle(fontSize: 14),
                              ),
                            )
                          : const Text('No SITREP loaded', style: TextStyle(fontSize: 14)),
                    ),
                    const SizedBox(height: 8),
                    if (_errorMessage != null)
                      Text(
                        _errorMessage!,
                        style: const TextStyle(color: Colors.red),
                      ),
                  ],
                ),
              ),
            ),

            // Bottom-right raised sign-out button
            Positioned(
              bottom: 20,
              right: 16,
              child: _buildSignOutButton(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPillButton({
    required String label,
    required VoidCallback onPressed,
  }) {
    return Material(
      elevation: 6,
      shadowColor: Colors.black26,
      borderRadius: BorderRadius.circular(30),
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(30),
        child: Container(
          width: _pillWidth,
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: _buttonBg,
            borderRadius: BorderRadius.circular(30),
          ),
          alignment: Alignment.center,
          child: Text(
            label,
            style: TextStyle(
              color: _buttonText,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSignOutButton() {
    return Material(
      elevation: 10,
      borderRadius: BorderRadius.circular(14),
      shadowColor: Colors.black26,
      child: InkWell(
        onTap: _signOut,
        borderRadius: BorderRadius.circular(14),
        child: Container(
          width: 64,
          height: 64,
          decoration: BoxDecoration(
            color: _buttonBg,
            borderRadius: BorderRadius.circular(14),
          ),
          child: const Icon(
            Icons.exit_to_app,
            size: 28,
          ),
        ),
      ),
    );
  }
}
