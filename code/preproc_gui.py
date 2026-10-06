# preproc_gui_arabic.py
# تطبيق PyQt6 لمعالجة الصور: تحميل -> اقتصاص -> تغيير حجم -> معالجة كاملة -> عرض المراحل -> حفظ الصورة النهائية

import sys
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QSlider, QMessageBox, QRubberBand
)
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QRect, QSize, QThread, pyqtSignal, QPoint

# -----------------------------
# تحويل مصفوفة RGB إلى QPixmap
# -----------------------------
def qpixmap_from_rgb_array(arr: np.ndarray, max_size: QSize = None) -> QPixmap:
    h, w = arr.shape[:2]
    bytes_per_line = 3 * w
    qimg = QImage(arr.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if max_size is not None:
        pix = pix.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pix

# -----------------------------
# عرض الصورة مع امكانية الاقتصاص
# -----------------------------
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = QPoint()
        self._selection_rect = QRect()
        self._is_selecting = False
        self.setMouseTracking(True)

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap
        self._rubber.hide()
        self._selection_rect = QRect()

    def mousePressEvent(self, event: QMouseEvent):
        if self.pixmap() is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_selecting = True
            self._origin = event.pos()
            self._rubber.setGeometry(QRect(self._origin, QSize()))
            self._rubber.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._is_selecting:
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self._rubber.setGeometry(rect)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._is_selecting:
            return
        self._is_selecting = False
        rect = self._rubber.geometry()
        self._selection_rect = rect

    def get_selection_image_coords(self, original_image: np.ndarray) -> QRect:
        if self._selection_rect.isNull() or self.pixmap() is None:
            return QRect()
        pm = self.pixmap()
        pm_rect = self.contentsRect()
        disp_w, disp_h = pm.width(), pm.height()
        x_off = (pm_rect.width() - disp_w) // 2
        y_off = (pm_rect.height() - disp_h) // 2
        sel = self._selection_rect.translated(-x_off, -y_off).intersected(QRect(0,0,disp_w,disp_h))
        if sel.isNull():
            return QRect()
        img_h, img_w = original_image.shape[:2]
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h
        x1 = int(sel.left() * scale_x)
        y1 = int(sel.top() * scale_y)
        x2 = int(sel.right() * scale_x)
        y2 = int(sel.bottom() * scale_y)
        x1, x2 = max(0, x1), min(img_w-1, x2)
        y1, y2 = max(0, y1), min(img_h-1, y2)
        return QRect(x1, y1, max(1, x2-x1), max(1, y2-y1))

    def clear_selection(self):
        self._rubber.hide()
        self._selection_rect = QRect()

# -----------------------------
# دوال معالجة الصور
# -----------------------------
TARGET_SIZE = (300, 300)

def center_crop_and_resize(img: np.ndarray, target=TARGET_SIZE, crop_rect: QRect = None):
    h, w = img.shape[:2]
    if crop_rect and not crop_rect.isNull():
        x, y, cw, ch = crop_rect.left(), crop_rect.top(), crop_rect.width(), crop_rect.height()
        x2 = min(x+cw, w); y2 = min(y+ch, h)
        cropped = img[y:y2, x:x2]
    else:
        m = min(h, w)
        sx = (w - m) // 2
        sy = (h - m) // 2
        cropped = img[sy:sy+m, sx:sx+m]
    resized = cv2.resize(cropped, target, interpolation=cv2.INTER_AREA)
    return resized

def denoise_bilateral(img: np.ndarray):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def apply_clahe_rgb(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def gray_world_correction(img: np.ndarray):
    imgf = img.astype(np.float32)
    avgR, avgG, avgB = imgf[:,:,0].mean(), imgf[:,:,1].mean(), imgf[:,:,2].mean()
    avgGray = (avgR + avgG + avgB) / 3.0
    imgf[:,:,0] = np.clip(imgf[:,:,0] * (avgGray / (avgR + 1e-8)), 0, 255)
    imgf[:,:,1] = np.clip(imgf[:,:,1] * (avgGray / (avgG + 1e-8)), 0, 255)
    imgf[:,:,2] = np.clip(imgf[:,:,2] * (avgGray / (avgB + 1e-8)), 0, 255)
    return imgf.astype(np.uint8)

def sharpen_image(img: np.ndarray):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def background_remove_otsu(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_rgb = cv2.merge([mask, mask, mask])
    return cv2.bitwise_and(img, mask_rgb)

def full_pipeline(img_orig: np.ndarray, crop_rect: QRect = None):
    stages = []
    img = img_orig.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    stages.append(img.copy())
    img_resized = center_crop_and_resize(img, TARGET_SIZE, crop_rect)
    stages.append(img_resized.copy())
    img_denoised = denoise_bilateral(img_resized)
    stages.append(img_denoised.copy())
    img_clahe = apply_clahe_rgb(img_denoised)
    stages.append(img_clahe.copy())
    img_color = gray_world_correction(img_clahe)
    stages.append(img_color.copy())
    img_sharp = sharpen_image(img_color)
    stages.append(img_sharp.copy())
    img_bg = background_remove_otsu(img_sharp)
    stages.append(img_bg.copy())
    return stages

# -----------------------------
# Thread لمعالجة الصورة بدون تجميد الواجهة
# -----------------------------
class PipelineWorker(QThread):
    finished_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(int, str)

    def __init__(self, img: np.ndarray, crop_rect: QRect = None):
        super().__init__()
        self.img = img
        self.crop_rect = crop_rect

    def run(self):
        stages = []
        img = self.img.copy()
        stages.append(img.copy())
        self.progress_signal.emit(0, "تم تحميل الصورة الأصلية")
        img_resized = center_crop_and_resize(img, TARGET_SIZE, self.crop_rect)
        stages.append(img_resized.copy())
        self.progress_signal.emit(1, "تم الاقتصاص وتغيير الحجم")
        img_denoised = denoise_bilateral(img_resized)
        stages.append(img_denoised.copy())
        self.progress_signal.emit(2, "تم إزالة الضوضاء")
        img_clahe = apply_clahe_rgb(img_denoised)
        stages.append(img_clahe.copy())
        self.progress_signal.emit(3, "تم تطبيق CLAHE")
        img_color = gray_world_correction(img_clahe)
        stages.append(img_color.copy())
        self.progress_signal.emit(4, "تم تصحيح الألوان")
        img_sharp = sharpen_image(img_color)
        stages.append(img_sharp.copy())
        self.progress_signal.emit(5, "تم زيادة الحدة")
        img_bg = background_remove_otsu(img_sharp)
        stages.append(img_bg.copy())
        self.progress_signal.emit(6, "تم إزالة الخلفية")
        self.finished_signal.emit(stages)

# -----------------------------
# الواجهة الرئيسية
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("معالجة الصور - PyQt6")
        self.resize(1000, 700)

        self.orig_img = None
        self.stages = []
        self.current_stage_idx = 0
        self.crop_rect_imgcoords = None

        self.image_label = ImageLabel()
        self.status_lbl = QLabel("قم بتحميل صورة للبدء.")
        self.btn_load = QPushButton("تحميل صورة")
        self.btn_select_crop = QPushButton("تحديد منطقة الاقتصاص")
        self.btn_apply_crop = QPushButton("تطبيق الاقتصاص وتغيير الحجم")
        self.btn_run_pipeline = QPushButton("تشغيل المعالجة الكاملة")
        self.btn_prev = QPushButton("السابق")
        self.btn_next = QPushButton("التالي")
        self.save_btn = QPushButton("حفظ الصورة النهائية")
        self.reset_btn = QPushButton("إعادة تعيين")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setEnabled(False)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label, stretch=1)
        left_layout.addWidget(self.status_lbl)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_load)
        controls.addWidget(self.btn_select_crop)
        controls.addWidget(self.btn_apply_crop)
        controls.addWidget(self.btn_run_pipeline)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.reset_btn)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.slider)
        nav.addWidget(self.btn_next)

        main_v = QVBoxLayout()
        main_v.addLayout(left_layout)
        main_v.addLayout(controls)
        main_v.addLayout(nav)

        container = QWidget()
        container.setLayout(main_v)
        self.setCentralWidget(container)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_select_crop.clicked.connect(self.enable_crop_mode)
        self.btn_apply_crop.clicked.connect(self.apply_crop_and_preview)
        self.btn_run_pipeline.clicked.connect(self.run_pipeline)
        self.btn_prev.clicked.connect(self.show_prev_stage)
        self.btn_next.clicked.connect(self.show_next_stage)
        self.save_btn.clicked.connect(self.save_final)
        self.reset_btn.clicked.connect(self.reset_all)
        self.slider.valueChanged.connect(self.slider_changed)

        self._crop_mode = False
        self.worker = None
        self.update_ui_state()

    def update_ui_state(self):
        has_img = self.orig_img is not None
        self.btn_select_crop.setEnabled(has_img)
        self.btn_apply_crop.setEnabled(has_img and self.image_label.get_selection_image_coords(self.orig_img).isValid())
        self.btn_run_pipeline.setEnabled(has_img)
        self.btn_prev.setEnabled(len(self.stages) > 0 and self.current_stage_idx > 0)
        self.btn_next.setEnabled(len(self.stages) > 0 and self.current_stage_idx < len(self.stages)-1)
        self.save_btn.setEnabled(len(self.stages) > 0)
        self.slider.setEnabled(len(self.stages) > 1)
        if len(self.stages) > 1:
            self.slider.setMaximum(len(self.stages)-1)
            self.slider.setValue(self.current_stage_idx)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "اختر صورة", str(Path.cwd()), "صور (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            QMessageBox.warning(self, "خطأ", "لا يمكن تحميل الصورة.")
            return
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        elif img_bgr.ndim == 3:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        self.orig_img = img_rgb
        pix = qpixmap_from_rgb_array(img_rgb, max_size=self.image_label.size())
        self.image_label.setPixmap(pix)
        self.status_lbl.setText("تم تحميل الصورة. يمكنك تحديد منطقة الاقتصاص أو تشغيل المعالجة.")
        self.stages = []
        self.current_stage_idx = 0
        self.crop_rect_imgcoords = None
        self.image_label.clear_selection()
        self.update_ui_state()

    def enable_crop_mode(self):
        if self.orig_img is None: return
        QMessageBox.information(self, "وضع الاقتصاص", "ارسم مستطيلًا على الصورة لتحديد منطقة الاقتصاص، ثم اضغط 'تطبيق الاقتصاص وتغيير الحجم'.")
        self._crop_mode = True

    def apply_crop_and_preview(self):
        if self.orig_img is None: return
        sel = self.image_label.get_selection_image_coords(self.orig_img)
        if sel.isNull():
            img_resized = center_crop_and_resize(self.orig_img, TARGET_SIZE, None)
        else:
            img_resized = center_crop_and_resize(self.orig_img, TARGET_SIZE, sel)
            self.crop_rect_imgcoords = sel
        self.stages = [self.orig_img, img_resized]
        self.current_stage_idx = 1
        self.display_current_stage()
        self.status_lbl.setText("تم تطبيق الاقتصاص وتغيير الحجم إلى 300×300.")
        self.update_ui_state()

    def run_pipeline(self):
        if self.orig_img is None: return
        self.btn_run_pipeline.setEnabled(False)
        self.status_lbl.setText("تشغيل المعالجة...")
        sel = self.image_label.get_selection_image_coords(self.orig_img)
        crop_rect = sel if sel.isValid() else None
        self.worker = PipelineWorker(self.orig_img, crop_rect)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.start()

    def on_progress(self, idx, msg):
        self.status_lbl.setText(f"المرحلة {idx}: {msg}")

    def on_pipeline_finished(self, stages):
        if not stages:
            QMessageBox.critical(self, "خطأ", "فشلت المعالجة.")
            self.status_lbl.setText("فشلت المعالجة.")
        else:
            self.stages = stages
            self.current_stage_idx = 0
            self.display_current_stage()
            self.status_lbl.setText("تمت المعالجة بنجاح.")
        self.worker = None
        self.update_ui_state()

    def display_current_stage(self):
        if not self.stages: return
        img = self.stages[self.current_stage_idx]
        pix = qpixmap_from_rgb_array(img, max_size=self.image_label.size())
        self.image_label.setPixmap(pix)
        self.status_lbl.setText(f"المرحلة {self.current_stage_idx} / {len(self.stages)-1}")
        self.update_ui_state()

    def show_prev_stage(self):
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self.display_current_stage()
            self.slider.setValue(self.current_stage_idx)

    def show_next_stage(self):
        if self.current_stage_idx < len(self.stages)-1:
            self.current_stage_idx += 1
            self.display_current_stage()
            self.slider.setValue(self.current_stage_idx)

    def slider_changed(self, v):
        if 0 <= v < len(self.stages):
            self.current_stage_idx = v
            self.display_current_stage()

    def save_final(self):
        if not self.stages: return
        final_img = self.stages[-1]
        path, _ = QFileDialog.getSaveFileName(self, "حفظ الصورة النهائية", str(Path.cwd()/"processed.png"), "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path: return
        bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, bgr)
        if ok:
            QMessageBox.information(self, "تم الحفظ", f"تم حفظ الصورة في:\n{path}")
        else:
            QMessageBox.warning(self, "خطأ", "فشل حفظ الصورة.")

    def reset_all(self):
        self.orig_img = None
        self.stages = []
        self.current_stage_idx = 0
        self.crop_rect_imgcoords = None
        self.image_label.clear_selection()
        self.image_label.setPixmap(QPixmap())
        self.status_lbl.setText("تمت إعادة التعيين. قم بتحميل صورة جديدة.")
        self.update_ui_state()

# -----------------------------
# تشغيل التطبيق
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
