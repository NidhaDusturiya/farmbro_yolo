# BISA JUGAAA PAKAI CONFIDENT
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

        # Ambil label, bounding box, dan confidence
        labels = results.names  
        boxes = results.boxes.xyxy.cpu().numpy() 
        confidences = results.boxes.conf.cpu().numpy()  
        class_ids = results.boxes.cls.cpu().numpy().astype(int) 

        # Konversi gambar ke format OpenCV (BGR)
        img_cv = np.array(img)  
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        CONFIDENCE_THRESHOLD = 0.60
        # Gambar bounding box di gambar
        detected_objects = []  # List untuk menyimpan hasil deteksi
        for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x_min, y_min, x_max, y_max = map(int, bbox)  
            label = labels[cls_id]  
            detected_objects.append({
                "label": label,
                "confidence": float(conf),  # Konversi ke float agar dapat dikirim sebagai JSON
                "bbox": bbox.tolist()  # Koordinat bounding box
            })

            # Gambar bounding box dengan label dan confidence
            cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)  
            cv2.putText(
                img_cv,
                f"{label} {conf:.2f}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, #font size
                (0, 255, 0),
                5, #ketebalan
                cv2.LINE_AA
            )

        # Konversi gambar dengan bounding box ke base64
        _, img_encoded = cv2.imencode('.jpg', img_cv)  # Encode image as JPG
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')  # Convert to base64 string

        # hasil respon
        result = {
            "count": len(detected_objects),  # Jumlah objek yang terdeteksi
            "detections": detected_objects,  # List hasil deteksi (label, confidence, bbox)
            "image": img_base64  # Kirim gambar dengan bounding box dalam format base64
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)