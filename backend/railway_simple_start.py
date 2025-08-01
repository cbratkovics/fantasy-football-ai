#!/usr/bin/env python3
"""Ultra-simple Railway startup - just get the PORT and run uvicorn"""
import os
import subprocess
import sys

# Get PORT from environment
port = os.environ.get('PORT', '8000')
print(f"Starting server on port {port}", flush=True)

# Run uvicorn with subprocess to avoid any import issues
cmd = [
    sys.executable, "-m", "uvicorn",
    "main:app",
    "--host", "0.0.0.0",
    "--port", port
]

print(f"Running: {' '.join(cmd)}", flush=True)
subprocess.run(cmd)