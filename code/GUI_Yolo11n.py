
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO

model = YOLO(r"models\best.pt")  
class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("WBC Detector")
        self.setGeometry(200, 200, 800, 600)

        self.image_label = QLabel("اختر صورة للكشف", self)
        self.image_label.setStyleSheet("font-size: 22px; text-align: center;")

        self.btn = QPushButton("اختيار صورة", self)
        self.btn.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image")

        if not file_path:
            return

        img = cv2.imread(file_path)

        results = model(img)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w

        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())
