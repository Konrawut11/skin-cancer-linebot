# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ติดตั้ง system dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git && \
    pip install -r yolov5/requirements.txt

# Copy project files
COPY . .

# ตั้งค่าพอร์ตที่ต้องการ (Railway จะตั้ง ENV PORT ให้อัตโนมัติ)
ENV PORT=5000

# Run app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
