// This file is used to bootstrap the Flutter web app.
// It loads the main Dart entrypoint and attaches the app to the DOM.

window.addEventListener('load', function() {
  // The Flutter web build outputs main.dart.js as the entrypoint.
  var script = document.createElement('script');
  script.src = 'main.dart.js';
  script.type = 'application/javascript';
  script.async = true;
  document.body.appendChild(script);
});