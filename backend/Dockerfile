# Sử dụng Python 3.10 mỏng nhẹ để tối ưu tài nguyên (Mục 13 - Ngân sách)
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho xử lý ảnh y tế
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# The local stack selects CPU wheels; GPU deployments select cu118.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url ${PYTORCH_INDEX_URL} --extra-index-url https://pypi.org/simple

# Copy danh sách thư viện và cài đặt
COPY requirements.txt requirements.constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements.constraints.txt

# Copy toàn bộ code vào container
COPY . .

# Chạy server với tính năng hot-reload để phát triển nhanh
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
