#ADA BOUNCDING BOX AJA BLM ADA CONFIDENCE
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

# Load model YOLOv8
model = YOLO("C:/FILE GUE/TUGAS TIF/SEMESTER 5/MBKM/image-segmentation-yolov8/image-segmentation-yolov8/yolov8-backend/models/best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Terima gambar yang diunggah
        img_file = request.files['image']
        img = Image.open(img_file.stream)

        # Deteksi menggunakan YOLOv8
        results = model(img)

        if isinstance(results, list):
            results = results[0]  # Ambil hasil pertama jika berupa list

        # Ambil label dan bounding box
        labels = results.names  # Nama kelas
        bboxes = results.boxes.xyxy.cpu().numpy()  # Koordinat bounding box (xyxy format)

        # Konversi gambar ke format OpenCV (BGR)
        img_cv = np.array(img)  # Convert to numpy array
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Gambar bounding box di gambar
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox)  # Ambil koordinat
            # Gambar bounding box (warna hijau dengan ketebalan 2)
            cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)

        # Konversi gambar dengan bounding box ke base64
        _, img_encoded = cv2.imencode('.jpg', img_cv)  # Encode image as JPG
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')  # Convert to base64 string

        # Persiapkan hasil respon
        result = {
            "count": len(bboxes),  # Jumlah objek yang terdeteksi (ayam)
            "bboxes": bboxes.tolist(),  # Koordinat bounding box
            "labels": labels,
            "image": img_base64  # Kirim gambar dengan bounding box dalam format base64
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
