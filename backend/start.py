#!/usr/bin/env python3
"""Railway-compatible startup script"""
import os
import sys

# FORCE immediate output
print("RAILWAY START.PY EXECUTING", file=sys.stderr, flush=True)
print("RAILWAY START.PY EXECUTING", file=sys.stdout, flush=True)

# Get port from environment
port = os.environ.get('PORT', '8000')
print(f"Using PORT: {port}", flush=True)

# Execute uvicorn with proper port
import subprocess
cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", port]
print(f"Executing: {' '.join(cmd)}", flush=True)
subprocess.run(cmd)