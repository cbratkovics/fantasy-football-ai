#!/usr/bin/env python3
"""Emergency minimal server that MUST work"""
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

print(f"Emergency server starting - Python {sys.version}", flush=True)
sys.stdout.flush()

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "healthy", 
                "server": "emergency",
                "timestamp": str(os.times()),
                "port": os.environ.get('PORT', 'unknown')
            })
            self.wfile.write(response.encode())
            print(f"Health check served successfully", flush=True)
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "message": "Fantasy Football AI - Emergency Mode",
                "status": "running"
            })
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)
        sys.stdout.flush()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting emergency server on 0.0.0.0:{port}", flush=True)
    sys.stdout.flush()
    
    try:
        server = HTTPServer(('0.0.0.0', port), HealthHandler)
        print(f"✓ Server listening on port {port}", flush=True)
        print(f"✓ Health endpoint: http://0.0.0.0:{port}/health", flush=True)
        sys.stdout.flush()
        server.serve_forever()
    except Exception as e:
        print(f"✗ Failed to start emergency server: {e}", flush=True)
        sys.stdout.flush()
        sys.exit(1)