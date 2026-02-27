FROM python:3.10-slim

# Install necessary system dependencies (especially for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user for Hugging Face Spaces (Spaces run as user 1000)
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install dependencies, plus gunicorn for production serving
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Command to run the application using gunicorn
# 120s timeout ensures the AI model has time to load and TTA runs without timing out
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "webapp.app:app"]
