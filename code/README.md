# 🔬 مصنف خلايا الدم البيضاء

واجهات مختلفة لتصنيف خلايا الدم البيضاء باستخدام النموذج المدرّب

## 📋 المحتويات

- `simple_classifier_interface.py` - واجهة رسومية بسيطة باستخدام Tkinter
- `classify_image_cli.py` - واجهة سطر الأوامر
- `web_classifier.py` - واجهة ويب باستخدام Flask
- `colab_classifier.py` - مصنف مبسط لـ Google Colab
- `colab_setup.py` - إعداد بيئة Google Colab
- `best_model.h5` - النموذج المدرّب (متوفر في الدليل الرئيسي)

## 🖥️ واجهة رسومية (Tkinter)

لتشغيل الواجهة الرسومية:

```bash
python simple_classifier_interface.py
```

### الميزات:
- واجهة رسومية سهلة الاستخدام
- عرض الصورة المختارة
- عرض نتائج التصنيف فوراً
- شريط تقدم أثناء التصنيف

## 💻 واجهة سطر الأوامر

لتشغيل الواجهة عبر سطر الأوامر:

```bash
python classify_image_cli.py path/to/image.jpg
```

### الميزات:
- بسيطة وسريعة
- مناسبة للدمج في سكريبتات
- عرض النتائج في الطرفية

## 🌐 واجهة ويب (Flask)

لتشغيل الواجهة عبر المتصفح:

1. تأكد من تثبيت المكتبات المطلوبة:
```bash
pip install -r requirements.txt
```

2. تشغيل الخادم:
```bash
python web_classifier.py
```

3. فتح المتصفح والذهاب إلى:
```
http://localhost:5000
```

### الميزات:
- واجهة ويب سهلة الاستخدام
- تصميم جذاب ومتجاوب
- دعم لعدة أنواع من الصور
- عرض نسبة الثقة بشكل مرئي

## ☁️ Google Colab

لتشغيل المشروع في Google Colab:

1. نفذ ملف الإعداد:
```python
!python colab_setup.py
```

2. اتبع التعليمات لتنزيل مجموعة البيانات

3. استخدم ملف `colab_classifier.py` للتصنيف

### الميزات:
- يعمل مباشرة في بيئة Colab
- لا يتطلب تثبيت محلي
- يدعم رفع الصور من الجهاز

## 📁 هيكل المجلدات

```
project/
├── best_model.h5          # النموذج المدرّب
├── simple_classifier_interface.py
├── classify_image_cli.py
├── web_classifier.py
├── colab_classifier.py
├── colab_setup.py
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   └── result.html
└── uploads/               # مجلد مؤقت للصور المحملة
```

## ⚙️ المتطلبات

- Python 3.7+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Flask (للوحة الويب)
- Tkinter (للوحة الرسومية - موجود عادة مع Python)

## 🎯 استخدام النموذج

جميع الواجهات تستخدم نفس النموذج المدرّب (`best_model.h5`) الذي تم إنشاؤه في مراحل سابقة من المشروع.

### فئات التصنيف:
1. Basophil
2. Banded Neutrophil
3. Eosinophil
4. Erythroblast
5. Lymphocyte
6. Meta-myelocyte
7. Monocyte
8. Myelocyte
9. Platelet
10. Pro-myelocyte
11. Segmented Neutrophil

## 🛠️ التطوير المستقبلي

- إضافة دعم لعدة صور في نفس الوقت
- تحسين دقة التصنيف
- إضافة خاصية حفظ النتائج
- دعم لغات متعددة