#!/usr/bin/env python3
"""
Railway-optimized startup script for Fantasy Football AI Backend
Handles environment diagnostics and graceful fallbacks
"""

import sys
import os

# IMMEDIATE output to confirm script is running
print("=" * 50, flush=True)
print("RAILWAY STARTUP SCRIPT BEGINNING", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Script: {__file__}", flush=True)
print(f"PID: {os.getpid()}", flush=True)
print(f"PORT: {os.environ.get('PORT', 'NOT SET')}", flush=True)
print(f"Working Dir: {os.getcwd()}", flush=True)
print(f"Files in dir: {os.listdir('.')}", flush=True)
print("=" * 50, flush=True)
sys.stdout.flush()

import logging
import time
from datetime import datetime

# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check critical environment variables and system status"""
    logger.info("=" * 80)
    logger.info("RAILWAY STARTUP DIAGNOSTICS")
    logger.info("=" * 80)
    
    # Check if we're actually in Railway
    railway_indicators = [
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_DEPLOYMENT_ID", 
        "RAILWAY_SERVICE_ID",
        "RAILWAY_REPLICA_ID"
    ]

    print("\nRailway Environment Check:", flush=True)
    for var in railway_indicators:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var}: {value}", flush=True)
    sys.stdout.flush()

    # Network binding test
    import socket
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('0.0.0.0', 0))
        test_socket.close()
        print("✓ Can bind to 0.0.0.0", flush=True)
    except Exception as e:
        print(f"✗ Cannot bind to 0.0.0.0: {e}", flush=True)
    sys.stdout.flush()
    
    # Basic system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    
    # Critical environment variables
    port = os.getenv('PORT')
    logger.info(f"PORT environment variable: {port}")
    if not port:
        logger.error("CRITICAL: PORT environment variable not set!")
        return False
    
    # Check if port is valid
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            logger.error(f"CRITICAL: Invalid port number: {port_int}")
            return False
        logger.info(f"Port validation: OK ({port_int})")
    except ValueError:
        logger.error(f"CRITICAL: PORT is not a valid integer: {port}")
        return False
    
    # Optional environment variables
    env_vars = ['DATABASE_URL', 'REDIS_URL', 'ENVIRONMENT', 'PYTHONPATH']
    for var in env_vars:
        value = os.getenv(var)
        status = "SET" if value else "NOT SET"
        logger.info(f"{var}: {status}")
    
    # Check file system
    critical_files = ['main.py', 'requirements.txt']
    for file in critical_files:
        if os.path.exists(file):
            logger.info(f"File check: {file} - EXISTS")
        else:
            logger.error(f"File check: {file} - MISSING")
            return False
    
    return True

def test_imports():
    """Test critical imports without full startup"""
    logger.info("Testing critical imports...")
    
    try:
        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        logger.error(f"FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        logger.info(f"Uvicorn import: OK")
    except ImportError as e:
        logger.error(f"Uvicorn import failed: {e}")
        return False
    
    try:
        # Test main app import
        from main import app
        logger.info("Main app import: OK")
        return True
    except Exception as e:
        logger.error(f"Main app import failed: {e}")
        logger.exception("Full traceback:")
        return False

def start_minimal_server():
    """Start minimal FastAPI server as fallback"""
    logger.info("Starting minimal fallback server...")
    
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Fantasy Football AI - Minimal Fallback")
    
    @app.get("/")
    async def root():
        return JSONResponse({
            "status": "running",
            "message": "Fantasy Football AI - Minimal Mode",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    @app.get("/health")  
    async def health():
        return JSONResponse({
            "status": "healthy",
            "mode": "minimal",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting minimal server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)

def start_main_server():
    """Start the main FastAPI application"""
    logger.info("Starting main FastAPI application...")
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting main server on 0.0.0.0:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        # Add timeout and worker settings for Railway
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )

def start_fallback_chain():
    """Try multiple server options in order of complexity"""
    servers = [
        ("Import Check", "python check_imports.py", False),
        ("Main App", start_main_server, True),
        ("Minimal FastAPI", start_minimal_server, True),
        ("Emergency HTTP", "python emergency_server.py", True),
        ("Basic Python HTTP", f"python -m http.server {os.environ.get('PORT', 8000)}", True)
    ]
    
    for name, command, is_server in servers:
        print(f"\n{'='*50}", flush=True)
        print(f"Attempting: {name}", flush=True)
        print(f"{'='*50}", flush=True)
        sys.stdout.flush()
        
        try:
            if callable(command):
                # It's a function
                command()
            else:
                # It's a shell command
                print(f"Executing: {command}", flush=True)
                sys.stdout.flush()
                exit_code = os.system(command)
                if exit_code != 0 and not is_server:
                    print(f"Command failed with exit code: {exit_code}", flush=True)
                    continue
            
            # If we get here and it's a server, it should be running
            if is_server:
                print(f"✓ {name} should be running", flush=True)
                sys.stdout.flush()
                # Server started, wait forever
                while True:
                    time.sleep(60)
                    
        except Exception as e:
            print(f"✗ {name} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            continue
    
    # If we get here, everything failed
    print("\n" + "="*50, flush=True)
    print("CRITICAL: All server options failed!", flush=True)
    print("="*50, flush=True)
    sys.stdout.flush()
    sys.exit(1)

def main():
    """Main startup function with progressive fallbacks"""
    try:
        # Step 1: Environment diagnostics
        if not check_environment():
            logger.error("Environment check failed! Starting fallback chain...")
            start_fallback_chain()
        
        # Step 2: Import testing
        if not test_imports():
            logger.warning("Main app imports failed, starting fallback chain...")
            start_fallback_chain()
        
        # Step 3: Try main application first
        logger.info("Attempting main application...")
        try:
            start_main_server()
        except Exception as e:
            logger.error(f"Main server failed: {e}")
            logger.info("Starting fallback chain...")
            start_fallback_chain()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.exception("Full traceback:")
        start_fallback_chain()

if __name__ == "__main__":
    main()