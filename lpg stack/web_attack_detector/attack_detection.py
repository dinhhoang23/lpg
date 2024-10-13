import tensorflow as tf
from keras.models import load_model
import numpy as np
import json
import time
import requests
import re
from urllib.parse import urlparse, parse_qs, unquote
from data_utils import Data
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đường dẫn tới file log web
LOG_FILE_PATH = '/var/log/apache2/access.log'  # Thay đổi nếu cần thiết

# Tải mô hình từ 'full_model.h5'
model = load_model('full_model.h5')
logger.info("Loaded model from 'full_model.h5'")

# Tải cấu hình từ 'config.json'
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("Config file not found")
    exit(1)
except json.JSONDecodeError:
    logger.error("Invalid JSON in config file")
    exit(1)

# Khởi tạo lớp Data từ data_utils.py
alphabet = config["data"]["alphabet"]
input_size = config["data"]["input_size"]
num_of_classes = config["data"]["num_of_classes"]

data_processor = Data(
    data_source=None,  # Không cần thiết trong trường hợp này
    alphabet=alphabet,
    input_size=input_size,
    num_of_classes=num_of_classes
)

# Ánh xạ số lớp thành tên loại tấn công
label_mapping = {
    0: 'normal',
    1: 'sqli',
    2: 'xss',
    3: 'sqli',
    4: 'sqli'
}

# Lấy URL Loki từ biến môi trường hoặc sử dụng giá trị mặc định
LOKI_URL = os.environ.get('LOKI_URL', 'http://loki:3100')

# Hàm theo dõi file log (giống như 'tail -f')
def follow_log(file_path):
    try:
        with open(file_path, 'r') as file:
            file.seek(0, 2)  # Di chuyển con trỏ tới cuối file
            while True:
                line = file.readline()
                if not line:
                    time.sleep(0.1)  # Chờ nếu không có dòng mới
                    continue
                yield line.strip()
    except IOError as e:
        logger.error(f"Error reading log file: {str(e)}")
        yield None

def extract_request(log_line):
    match = re.search(r'\"(GET|POST|PUT|DELETE|HEAD|OPTIONS) (.+?) HTTP\/', log_line)
    if match:
        method = match.group(1)
        request = match.group(2)
        return method, request
    else:
        return None, None

def extract_payloads_from_request(request):
    parsed_url = urlparse(request)
    query_params = parse_qs(parsed_url.query)
    payloads = []
    for key, values in query_params.items():
        for value in values:
            decoded_value = unquote(value)
            payloads.append(decoded_value)
    return payloads

def extract_payloads(log_line):
    method, request = extract_request(log_line)
    if request:
        payloads = extract_payloads_from_request(request)
        return payloads
    else:
        return []

def preprocess_payload(payload):
    if payload is None or len(payload) > input_size:
        return None
    encoded_payload = data_processor.str_to_indexes(payload)
    return np.array(encoded_payload)

# Hàm dự đoán loại tấn công
def predict_attack_type(log_line, threshold=0.5):
    payloads = extract_payloads(log_line)
    for payload in payloads:
        processed_payload = preprocess_payload(payload)
        if processed_payload is None:
            continue
        processed_payload = processed_payload.reshape(1, -1)
        prediction = model.predict(processed_payload, verbose=0)
        
        predicted_class = np.argmax(prediction[0])  # Lấy chỉ số lớp dự đoán có xác suất cao nhất
        attack_prob_value = prediction[0][predicted_class]  # Xác suất của lớp được dự đoán
        
        if attack_prob_value >= threshold:
            if predicted_class != 0:  # Lớp 0 là 'normal'
                return predicted_class  # Trả về lớp tấn công được dự đoán
            else:
                continue  # Tiếp tục kiểm tra payload tiếp theo
    return 0  # Trả về 0 nếu không phát hiện tấn công

def push_log_to_loki(log_line, log_type, attack_type=None):
    try:
        headers = {'Content-type': 'application/json'}
        labels = {
            'job': 'web_attack_detection',
            'level': 'alert' if log_type == "attack" else 'info',
            'type': log_type
        }
        if attack_type:
            labels['attack_type'] = attack_type  # Thêm loại tấn công vào labels
        data = {
            'streams': [
                {
                    'stream': labels,
                    'values': [
                        [str(int(time.time() * 1e9)), log_line]
                    ]
                }
            ]
        }
        response = requests.post(f'{LOKI_URL}/loki/api/v1/push', data=json.dumps(data), headers=headers)
        if response.status_code != 204:
            logger.error(f"Failed to send log to Loki: {response.text}")
        return response.status_code
    except requests.RequestException as e:
        logger.error(f"Error sending log to Loki: {str(e)}")
        return None

def process_log_line(log_line):
    predicted_class = predict_attack_type(log_line)
    predicted_class = int(predicted_class)  # Chuyển đổi sang kiểu int
    attack_type = label_mapping.get(predicted_class, 'unknown')
    if predicted_class != 0:  # Nếu không phải lớp 'normal'
        logger.warning(f"[ATTACK DETECTED] {log_line} - Loại tấn công: {attack_type}")
        push_log_to_loki(log_line, "attack", attack_type)
    else:
        logger.info(f"[NORMAL] {log_line}- Normal")
        push_log_to_loki(log_line, "normal",attack_type)

def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            for log_line in follow_log(LOG_FILE_PATH):
                if log_line is not None:
                    executor.submit(process_log_line, log_line)
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()