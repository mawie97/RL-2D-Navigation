# Use Python 3.10 (matches your environment.yml)
FROM python:3.10-slim

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# System packages needed by mujoco, OpenGL stuff, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /workspace

# Copy only requirements first (better build caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project into the image
COPY . .

# Make sure your project is importable
ENV PYTHONPATH=/workspace

# Default command when the container runs
# >>> CHANGE THIS to the actual entry script you use <<<
CMD ["python", "navppo/run_train_env.py"]
