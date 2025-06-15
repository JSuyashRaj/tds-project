FROM python:3.10-slim

# Install system tools, Rust and Python build tools
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    cargo \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Launch FastAPI using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
