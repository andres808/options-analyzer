import http.server
import socketserver
import os
import webbrowser
from threading import Timer

# Set port for the server
PORT = 5001

class Handler(http.server.SimpleHTTPRequestHandler):
    # Ensure proper MIME types for the static site
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Add proper CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

def open_browser():
    """Open browser to the local server"""
    webbrowser.open(f'http://localhost:{PORT}')

# Start the server
def main():
    print(f"Starting local server on port {PORT}...")
    print(f"Navigate to: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    # Open browser after a short delay
    Timer(1.0, open_browser).start()
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()