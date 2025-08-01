#!/bin/bash
# Test Railway deployment locally

echo "=========================================="
echo "Testing Railway Deployment Locally"
echo "=========================================="

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker stop railway-test-container 2>/dev/null
docker rm railway-test-container 2>/dev/null

# Build the image
echo -e "\nBuilding Docker image..."
docker build -t railway-test . || {
    echo "Build failed!"
    exit 1
}

# Run the container with Railway-like environment
echo -e "\nRunning container with Railway environment..."
docker run -d \
  -p 8000:8000 \
  -e PORT=8000 \
  -e RAILWAY_ENVIRONMENT=production \
  -e PYTHONUNBUFFERED=1 \
  --name railway-test-container \
  railway-test

# Give it a moment to start
echo -e "\nWaiting for container to start..."
sleep 3

# Check if container is still running
if docker ps | grep railway-test-container > /dev/null; then
    echo "✓ Container is running"
else
    echo "✗ Container crashed! Checking logs..."
    docker logs railway-test-container
    exit 1
fi

# Show logs
echo -e "\n========== CONTAINER LOGS =========="
docker logs railway-test-container

# Test health endpoint
echo -e "\n========== TESTING ENDPOINTS =========="
echo "Testing /health endpoint..."
curl -f http://localhost:8000/health || echo "Health check failed"

echo -e "\nTesting / endpoint..."
curl -f http://localhost:8000/ || echo "Root endpoint failed"

# Keep showing logs
echo -e "\n========== FOLLOWING LOGS =========="
echo "Press Ctrl+C to stop..."
docker logs -f railway-test-container