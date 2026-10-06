import os

folders = [
    "datasat\Validate\Banded Neutrophil",
    "datasat\Validate\Basophil",
    "datasat\Validate\Eosinophil",
    "datasat\Validate\Erythroblast",
    "datasat\Validate\Lymphocyte",
    "datasat\Validate\Meta-myelocyte",
    "datasat\Validate\Monocyte",
    "datasat\Validate\Myelocyte",
    "datasat\Validate\Platelet",
    "datasat\Validate\Pro-myelocyte",
    "datasat\Validate\Segmented Neutrophil",
]

IMAGE_EXTENSIONS = {'.jpg', '.png', '.webp'}

def clean_label(name: str) -> str:
    """تنظيف اسم التصنيف ليكون مناسبًا كاسم ملف.
    - استبدال الفراغات بـ _
    - الإبقاء على الأحرف والأرقام و _ و - فقط
    """
    name = name.replace(' ', '_')
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in ('_', '-'):
            allowed.append(ch)
        else:
            allowed.append('_')
    return ''.join(allowed)

def rename_images_sequentially():
    for folder in folders:
        if not os.path.exists(folder):
            print(f"❌ المجلد غير موجود: {folder}")
            continue
        
        # استخراج اسم التصنيف من اسم المجلد وتنظيفه
        label = clean_label(os.path.basename(folder))
        print(f"\n🔄 معالجة المجلد: {folder} (التصنيف: {label})")
        
        # جلب جميع الملفات وفلترة الصور
        all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        image_files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        
        if not image_files:
            print("   ⚠️ لا توجد صور في هذا المجلد")
            continue
        
        # ترتيب الملفات أبجديًا لتجنب العشوائية
        image_files.sort()
        total_files = len(image_files)
        
        # المرحلة 1: إعادة التسمية المؤقتة (لتجنب التعارض)
        temp_names = []
        for idx, filename in enumerate(image_files, 1):
            ext = os.path.splitext(filename)[1].lower()
            temp_name = f"__temp__{idx:04d}{ext}"  # __temp__0001.jpg
            old_path = os.path.join(folder, filename)
            temp_path = os.path.join(folder, temp_name)
            
            try:
                os.rename(old_path, temp_path)
                temp_names.append(temp_name)
                print(f"   ➡️  المرحلة 1: {filename} → {temp_name}")
            except Exception as e:
                print(f"   ❌ خطأ في المرحلة 1: {filename} - {str(e)}")
        
        # المرحلة 2: إعادة التسمية النهائية
        temp_names.sort()  # ضمان الترتيب الصحيح
        for idx, temp_name in enumerate(temp_names, 1):
            ext = os.path.splitext(temp_name)[1].lower()
            new_name = f"{label}_{idx:03d}{ext}"  # Basophil_001.jpg
            temp_path = os.path.join(folder, temp_name)
            new_path = os.path.join(folder, new_name)
            
            try:
                os.rename(temp_path, new_path)
                print(f"   ✅ المرحلة 2: {temp_name} → {new_name}")
            except Exception as e:
                print(f"   ❌ خطأ في المرحلة 2: {temp_name} - {str(e)}")
        
        print(f"   ✔️  اكتملت معالجة {total_files} صورة")

if __name__ == "__main__":
    print("🚀 بدء عملية إعادة تسمية الصور...")
    rename_images_sequentially()
    print("\n✨ العملية اكتملت بنجاح!")