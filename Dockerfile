# Use a deep learning base image (PyTorch example)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Uninstall Conda to prevent conflicts
RUN rm -rf /opt/conda

# Install necessary system dependencies, including Python3-pip
RUN apt-get update && apt-get install -y git wget unzip python3-pip && ln -s /usr/bin/python3 /usr/bin/python && rm -rf /var/lib/apt/lists/*

# Create a working directory for your project
WORKDIR /app

# Install Poetry using pip
RUN pip install --no-cache-dir poetry

# Disable Poetry virtualenv creation (use system Python)
RUN poetry config virtualenvs.create false

# Copy dependency files first (to leverage Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of the code
COPY . .