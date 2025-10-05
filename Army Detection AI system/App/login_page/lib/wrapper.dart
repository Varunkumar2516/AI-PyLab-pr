import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

// Import the correct class names
import 'homepage.dart'; // This defines Homepage
import 'login.dart';    // This defines Login

class Wrapper extends StatefulWidget {
  const Wrapper({super.key});

  @override
  State<Wrapper> createState() => _WrapperState();
}

class _WrapperState extends State<Wrapper> {
  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.active) {
          if (snapshot.hasData) {
            return const Homepage(); // Use correct class name
          } else {
            return const Login();    // Use correct class name
          }
        }
        // Show loading indicator while waiting for auth state
        return const Scaffold(
          body: Center(child: CircularProgressIndicator()),
        );
      },
    );
  }
}
