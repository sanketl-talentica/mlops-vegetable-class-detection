FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by torchvision (libGL for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project package
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["python", "application.py"]
