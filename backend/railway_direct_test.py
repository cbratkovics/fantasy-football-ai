#!/usr/bin/env python3
"""Direct Railway test - absolute minimal server that MUST work"""

# NO IMPORTS except standard library to avoid any issues
import os
import sys

# Immediate output
print("RAILWAY_DIRECT_TEST: Starting", flush=True)
print(f"RAILWAY_DIRECT_TEST: Python {sys.version}", flush=True)
print(f"RAILWAY_DIRECT_TEST: PORT={os.environ.get('PORT', 'NOT SET')}", flush=True)
sys.stdout.flush()

# Try the absolute simplest HTTP server possible
try:
    port = int(os.environ.get('PORT', 8000))
    print(f"RAILWAY_DIRECT_TEST: Attempting basic HTTP server on port {port}", flush=True)
    sys.stdout.flush()
    
    # Use the most basic HTTP server
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    
    class DirectHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
                print("RAILWAY_DIRECT_TEST: Health check served", flush=True)
            else:
                super().do_GET()
    
    server = HTTPServer(('0.0.0.0', port), DirectHandler)
    print(f"RAILWAY_DIRECT_TEST: Server started on 0.0.0.0:{port}", flush=True)
    sys.stdout.flush()
    server.serve_forever()
    
except Exception as e:
    print(f"RAILWAY_DIRECT_TEST: FAILED - {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)