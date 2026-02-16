FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="go2432@wayne.edu"
LABEL description="LSTV Fusion Pipeline - Combines SPINEPS + Uncertainty Detection"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Default command
CMD ["/bin/bash"]
