#!/usr/bin/env python3
"""Test all imports before server starts"""
import sys

print("=" * 50, flush=True)
print("IMPORT CHECK STARTING", flush=True)
print(f"Python: {sys.version}", flush=True)
print("=" * 50, flush=True)
sys.stdout.flush()

imports_to_test = [
    "os",
    "sys",
    "json",
    "logging",
    "datetime",
    "fastapi",
    "uvicorn",
    "pydantic",
    "dotenv",
    "sqlalchemy",
    "psycopg2",
    "asyncpg",
    "redis",
    "stripe",
    "jose",
    "passlib",
    "httpx",
    "celery",
    "aiohttp",
]

failed = []
warnings = []

for module in imports_to_test:
    try:
        __import__(module)
        print(f"✓ {module}", flush=True)
    except ImportError as e:
        if module in ["sqlalchemy", "psycopg2", "redis", "stripe"]:
            print(f"⚠ {module}: {e} (optional)", flush=True)
            warnings.append((module, str(e)))
        else:
            print(f"✗ {module}: {e}", flush=True)
            failed.append((module, str(e)))
    sys.stdout.flush()

print("\n" + "=" * 50, flush=True)
if failed:
    print(f"CRITICAL: Failed imports: {len(failed)}", flush=True)
    for module, error in failed:
        print(f"  - {module}: {error}", flush=True)
    sys.exit(1)
elif warnings:
    print(f"WARNING: Optional imports failed: {len(warnings)}", flush=True)
    for module, error in warnings:
        print(f"  - {module}: {error}", flush=True)
    print("\nApp should still run with reduced functionality", flush=True)
else:
    print("SUCCESS: All imports working!", flush=True)

sys.stdout.flush()