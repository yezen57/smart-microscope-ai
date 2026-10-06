# gui_wbc.py
# واجهة PyQt6 لعرض صورة -> توقع نموذج EfficientNetB3
# يدعم: تحميل صورة من القرص، التقاط من كاميرا Pi (إن وُجدت)، عرض أعلى 5 احتمالات ورسم بياني (Plotly).

import sys, json, time
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

# محاولة استيراد Picamera2 (اختياري للـ Raspberry Pi)
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except Exception:
    CAMERA_AVAILABLE = False

# -------- إعداد المسارات والإعدادات -----------
MODEL_PATH = r"models\best_modelt.keras"   # غيّر إن لزم
CLASSN_PATH = "class_names.json"  # الملف الذي طلبت أن يحتوي الرقم 11
IMG_SIZE = (224, 224)
PLOT_FILE = "prob_plot.png"

# -------- تحميل النموذج -----------
if not Path(MODEL_PATH).exists():
    print(f"⚠️ نموذج غير موجود في المسار: {MODEL_PATH}")
    # لا نُنهِي البرنامج هنا؛ سنحاول لاحقًا تحميله أثناء التنبؤ إذا رغبت.
model = tf.keras.models.load_model(MODEL_PATH)

# -------- تحميل أسماء الفئات (مرن للتعامل مع رقم بدل قائمة) -----------
def load_class_names(path):
    p = Path(path)
    if not p.exists():
        print(f"⚠️ {path} غير موجود. سيتم توليد أسماء افتراضية حسب مخرجات النموذج.")
        n = model.output_shape[-1] if model is not None else 11
        return [f"Class_{i}" for i in range(n)]
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print("خطأ في قراءة JSON:", e)
        n = model.output_shape[-1] if model is not None else 11
        return [f"Class_{i}" for i in range(n)]
    # حالتان: القائمة أو رقم (أو نص رقمي)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, int):
        return [f"Class_{i}" for i in range(raw)]
    if isinstance(raw, str) and raw.strip().isdigit():
        n = int(raw.strip())
        return [f"Class_{i}" for i in range(n)]
    # fallback: حاول استخدام شكل إخراج النموذج
    if model is not None:
        n = model.output_shape[-1]
        return [f"Class_{i}" for i in range(n)]
    return []

class_names = load_class_names(CLASSN_PATH)
print("Loaded/Generated class names count:", len(class_names))

# -------- دوال المعالجة والتنبؤ -----------
def preprocess_image_for_model(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)  # [-1,1]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(path, top_k=5):
    # يتأكد أن النموذج محمل
    global model
    if model is None:
        raise RuntimeError("Model not loaded.")
    x = preprocess_image_for_model(path)
    preds = model.predict(x, verbose=0)
    # preds قد تكون (1, N) أو (N,) — نأخذ العنصر الأول إن كانت batch
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
            probs = preds[0]
        else:
            probs = preds.flatten()
    else:
        # في حالة إرجاع TF Tensor
        probs = np.array(preds)[0] if hasattr(preds, '__iter__') else np.array(preds).flatten()
    probs = probs.astype(float)
    # تأكد أن طول class_names يطابق طول الاحتمالات؛ إن لم يكن، قم بتعديل class_names مؤقتًا
    if len(class_names) != probs.shape[0]:
        print("تحذير: عدد أسماء الفئات لا يطابق مخرجات النموذج. تعديل افتراضي مؤقت.")
        # توليد أسماء مؤقتة
        tmp_names = [f"Class_{i}" for i in range(probs.shape[0])]
    else:
        tmp_names = class_names

    idxs = probs.argsort()[::-1][:top_k]
    top = [(tmp_names[i], float(probs[i])) for i in idxs]

    # رسم الاحتمالات (top_k) وحفظه كصورة PNG باستخدام kaleido إن أمكن
    labels = [tmp_names[i] for i in idxs]
    values = [float(probs[i] * 100.0) for i in idxs]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color='steelblue'))
    fig.update_layout(title="Prediction Probabilities (%)", yaxis=dict(title="%"), xaxis_tickangle=-35, height=420, margin=dict(t=50,b=100))
    try:
        # تحتاج kaleido مثبت: pip install -U kaleido
        fig.write_image(PLOT_FILE)
    except Exception as e:
        # فشل في حفظ الصورة (ربما kaleido غير مثبت) -> احفظ كـ HTML بديل
        html_file = "prob_plot.html"
        fig.write_html(html_file)
        print("لم يتم حفظ PNG (kaleido مفقود). حفظ HTML بديل:", html_file)

    return top

# -------- واجهة PyQt6 -----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Digital Microscope - WBC Classifier")
        self.setMinimumSize(920, 820)

        # عناصر واجهة المستخدم
        self.img_label = QtWidgets.QLabel("No Image")
        self.img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_label.setFixedSize(640, 480)
        self.img_label.setStyleSheet("border: 1px solid #ccc;")

        self.plot_label = QtWidgets.QLabel("Probabilities Plot")
        self.plot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setFixedSize(640, 280)
        self.plot_label.setStyleSheet("border: 1px solid #ccc;")

        self.load_btn = QtWidgets.QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.capture_btn = QtWidgets.QPushButton("Capture from Camera")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(CAMERA_AVAILABLE)

        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.predict_btn.clicked.connect(self.on_predict)
        self.predict_btn.setEnabled(False)

        self.result_box = QtWidgets.QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(140)

        # Layouts
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.predict_btn)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.img_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.plot_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_box)
        self.setLayout(main_layout)

        self.current_image_path = None

        # camera init (if available)
        if CAMERA_AVAILABLE:
            try:
                self.cam = Picamera2()
                self.cam.configure(self.cam.create_still_configuration(main={"format":"RGB888"}))
                self.cam.start()
            except Exception as e:
                print("تعذر تهيئة الكاميرا:", e)
                self.capture_btn.setEnabled(False)

    def load_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fname:
            return
        self.current_image_path = fname
        pix = QtGui.QPixmap(fname)
        pix = pix.scaled(self.img_label.width(), self.img_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.img_label.setPixmap(pix)
        self.result_box.clear()
        self.predict_btn.setEnabled(True)

    def capture_image(self):
        if not CAMERA_AVAILABLE:
            self.result_box.setPlainText("Camera not available on this device.")
            return
        try:
            save_path = "captured.jpg"
            self.cam.capture_file(save_path)
            self.current_image_path = save_path
            pix = QtGui.QPixmap(save_path)
            pix = pix.scaled(self.img_label.width(), self.img_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.img_label.setPixmap(pix)
            self.predict_btn.setEnabled(True)
        except Exception as e:
            self.result_box.setPlainText("Camera capture error:\n" + str(e))

    def on_predict(self):
        if not self.current_image_path:
            return
        self.result_box.setPlainText("Predicting... please wait.")
        QtWidgets.QApplication.processEvents()
        try:
            preds = predict_image(self.current_image_path)
            text = "Top predictions:\n\n"
            for label, prob in preds:
                text += f"{label:30s}: {prob*100:.2f}%\n"
            self.result_box.setPlainText(text)
            # load plot png if exists
            if Path(PLOT_FILE).exists():
                pix = QtGui.QPixmap(PLOT_FILE)
                pix = pix.scaled(self.plot_label.width(), self.plot_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.plot_label.setPixmap(pix)
            else:
                self.plot_label.setText("Probabilities plot not available (kaleido may be missing).")
        except Exception as e:
            self.result_box.setPlainText(f"Error during prediction:\n{e}")

# -------- تشغيل التطبيق -----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
