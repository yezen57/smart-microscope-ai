#!/usr/bin/env python3
# batch_preprocess.py
# يطبق سلسلة المعالجة على كل الصور داخل datasat orginal/Train
# يحفظ النتائج في datasat orginal/Train_processed (قابل للتعديل)

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import os

# -----------------------
# إعدادات (عدّل هنا إذا رغبت)
# -----------------------
INPUT_DIR = Path("datasat orginal/Train")       # مجلد الصور الأصلي
OUTPUT_DIR = Path("datasat orginal/Train_processed")  # مجلد الحفظ للصور المعالجة
TARGET_SIZE = (300, 300)   # width, height
OVERWRITE = False          # لو True سيحفظ فوق الملفات الأصلية (احذر)
SAVE_STAGES = False        # لو True سيحفظ صور كل مرحلة في مجلد فرعي stages/<stage_name>/
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -----------------------
# دوال المعالجة (نفسها كما في الواجهة)
# -----------------------
def center_crop_and_resize(img: np.ndarray, target=TARGET_SIZE, crop_rect=None):
    h, w = img.shape[:2]
    if crop_rect is not None:
        x, y, cw, ch = crop_rect
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
    # تجنب القسمة على صفر بإضافة epsilon
    eps = 1e-8
    imgf[:,:,0] = np.clip(imgf[:,:,0] * (avgGray / (avgR + eps)), 0, 255)
    imgf[:,:,1] = np.clip(imgf[:,:,1] * (avgGray / (avgG + eps)), 0, 255)
    imgf[:,:,2] = np.clip(imgf[:,:,2] * (avgGray / (avgB + eps)), 0, 255)
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

def run_pipeline_on_image(img_rgb: np.ndarray, crop_rect=None, return_stages=False):
    stages = []
    img = img_rgb.copy()
    stages.append(img.copy())  # original
    # crop & resize
    img_resized = center_crop_and_resize(img, TARGET_SIZE, crop_rect)
    stages.append(img_resized.copy())
    # denoise
    img_denoised = denoise_bilateral(img_resized)
    stages.append(img_denoised.copy())
    # clahe
    img_clahe = apply_clahe_rgb(img_denoised)
    stages.append(img_clahe.copy())
    # color correct
    img_color = gray_world_correction(img_clahe)
    stages.append(img_color.copy())
    # sharpen
    img_sharp = sharpen_image(img_color)
    stages.append(img_sharp.copy())
    # background removal
    img_bg = background_remove_otsu(img_sharp)
    stages.append(img_bg.copy())

    if return_stages:
        return stages
    else:
        return stages[-1]

# -----------------------
# إنشاء مجلدات الإخراج
# -----------------------
if not INPUT_DIR.exists():
    raise SystemExit(f"مسار الإدخال غير موجود: {INPUT_DIR}")

if OVERWRITE:
    out_root = INPUT_DIR
else:
    out_root = OUTPUT_DIR
out_root.mkdir(parents=True, exist_ok=True)

# إذا حفظنا المراحل الوسيطة، أنشئ المجلدات الفرعية
stage_names = ["original", "resized", "denoised", "clahe", "color_corrected", "sharpened", "bg_removed"]
if SAVE_STAGES:
    for s in stage_names:
        (out_root / "stages" / s).mkdir(parents=True, exist_ok=True)

# ملف لوج للأخطاء
error_log_path = out_root / "processing_errors.csv"
error_log = open(error_log_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(error_log)
csv_writer.writerow(["image_path", "error"])

# -----------------------
# معالجة كل الصور
# -----------------------
total = 0
processed = 0
failed = 0

# اجمع كل ملفات الصور داخل كل فئة
all_image_paths = []
for class_dir in sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()]):
    for p in class_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            all_image_paths.append(p)

print(f"Found {len(all_image_paths)} images to process.")
for img_path in tqdm(all_image_paths, desc="Processing images"):
    total += 1
    rel = img_path.relative_to(INPUT_DIR)
    out_path = (out_root / rel) if not OVERWRITE else img_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # قراءة الصورة
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError("cv2.imread returned None (file may be corrupted or unreadable)")

        # تحويل إلى RGB (مع دعم الشفافية)
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        elif img_bgr.ndim == 3:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)

        if SAVE_STAGES:
            stages = run_pipeline_on_image(img_rgb, crop_rect=None, return_stages=True)
            # حفظ كل مرحلة
            for idx, s_img in enumerate(stages):
                stage_name = stage_names[idx]
                save_dir = out_root / "stages" / stage_name / rel.parent.name
                save_dir.mkdir(parents=True, exist_ok=True)
                out_file = save_dir / img_path.name
                # تحويل RGB -> BGR قبل الحفظ
                cv2.imwrite(str(out_file), cv2.cvtColor(s_img, cv2.COLOR_RGB2BGR))
            # وحفظ النهاية أيضاً في المسار النهائي
            final_img = stages[-1]
            cv2.imwrite(str(out_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        else:
            final_img = run_pipeline_on_image(img_rgb, crop_rect=None, return_stages=False)
            cv2.imwrite(str(out_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        processed += 1
    except Exception as e:
        failed += 1
        csv_writer.writerow([str(img_path), str(e)])
        # استمر بالمجموعة التالية

error_log.close()

# -----------------------
# خلاصة
# -----------------------
print("---------------")
print(f"Total images: {total}")
print(f"Processed: {processed}")
print(f"Failed: {failed}")
print(f"Output root: {out_root.resolve()}")
print(f"Error log: {error_log_path.resolve()}")
