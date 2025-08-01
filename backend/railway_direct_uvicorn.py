#!/usr/bin/env python3
"""Direct uvicorn startup for Railway - handles PORT properly"""
import os
import sys

# Immediate output
print("=" * 50, flush=True)
print("RAILWAY DIRECT UVICORN STARTUP", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"PORT: {os.environ.get('PORT', 'NOT SET')}", flush=True)
print("=" * 50, flush=True)
sys.stdout.flush()

try:
    # Get port with proper error handling
    port_str = os.environ.get('PORT')
    if not port_str:
        print("ERROR: PORT environment variable not set!", flush=True)
        sys.exit(1)
    
    try:
        port = int(port_str)
        print(f"Port parsed successfully: {port}", flush=True)
    except ValueError:
        print(f"ERROR: PORT '{port_str}' is not a valid integer!", flush=True)
        sys.exit(1)
    
    # Import and run uvicorn
    print("Importing uvicorn...", flush=True)
    import uvicorn
    
    print(f"Starting uvicorn on 0.0.0.0:{port}...", flush=True)
    sys.stdout.flush()
    
    # Run with explicit integer port
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,  # This is now guaranteed to be an integer
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    print(f"STARTUP FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)