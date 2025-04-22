import http.server
import socketserver
import os
import webbrowser
import socket
from threading import Timer
import time

# Try different ports if the default is in use
DEFAULT_PORT = 8080
MAX_PORT_ATTEMPTS = 10

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

def open_browser(port):
    """Open browser to the local server"""
    webbrowser.open(f'http://localhost:{port}')

def find_available_port(start_port=DEFAULT_PORT, max_attempts=MAX_PORT_ATTEMPTS):
    """Find an available port starting from start_port"""
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        try:
            # Try to create a socket with the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            print(f"Port {port} is in use, trying next port...")
            continue
    
    # If we've tried all ports and none are available
    print(f"Couldn't find an available port after {max_attempts} attempts.")
    print(f"Please manually check for an available port and update the script.")
    return None

# Start the server
def main():
    # Find an available port
    port = find_available_port()
    
    if not port:
        return
    
    print(f"Starting local server on port {port}...")
    print(f"Navigate to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Make sure we're in the static_site directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    try:
        # Open browser after a short delay
        Timer(1.0, lambda: open_browser(port)).start()
        
        # Create the server with the found port
        with socketserver.TCPServer(("", port), Handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Try running the server manually with a different port.")

if __name__ == "__main__":
    main()