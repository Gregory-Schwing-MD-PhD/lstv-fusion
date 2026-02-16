#!/bin/bash
# Build and push LSTV Fusion Docker container

set -euo pipefail

DOCKER_USERNAME="go2432"
IMAGE_NAME="lstv-fusion"
TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building ${FULL_IMAGE}..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Build
docker build -t "${FULL_IMAGE}" -f docker/Dockerfile .

echo "✓ Build complete"
echo ""
echo "To push: docker push ${FULL_IMAGE}"
echo "To use in Singularity: singularity pull lstv-fusion.sif docker://${FULL_IMAGE}"
