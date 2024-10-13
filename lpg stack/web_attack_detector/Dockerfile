# Dockerfile

FROM python:3.9-slim

# Cài đặt các thư viện cần thiết


# Tạo thư mục ứng dụng
WORKDIR /app
COPY requirements.txt /app/
# Sao chép mã nguồn vào container
COPY attack_detection.py /app/
COPY data_utils.py /app/
COPY config.json /app/
COPY full_model.h5 /app/

# Chạy ứng dụng
CMD ["sh", "-c", "pip install -r requirements.txt && python attack_detection.py"]