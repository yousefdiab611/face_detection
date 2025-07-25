<<<<<<< HEAD
# نظام كشف وتتبع الوجوه المحسن

نظام متكامل للكشف عن الوجوه وتتبعها مع ميزات متقدمة باستخدام Python و OpenCV.

## الميزات الرئيسية
- كشف الوجوه في الوقت الفعلي
- تتبع الوجوه
- التعرف على الوجوه المعروفة
- اكتشاف الابتسامات
- اكتشاف العيون
- إحصائيات لحظية
- تنبيهات صوتية
- حفظ صور الوجوه
- واجهة مستخدم تفاعلية

## المتطلبات
- Python 3.11 أو أحدث
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- Face Recognition
- Playsound

## التثبيت
1. تثبيت المكتبات المطلوبة:
```bash
pip install -r requirements.txt
```

2. تشغيل البرنامج:
```bash
python enhanced_face_detection.py
```

## استخدام البرنامج
1. عند تشغيل البرنامج، سيظهر نافذة فيديو تظهر الكاميرا
2. سيتم كشف الوجوه تلقائيًا ورسم مربعات خضراء حولها
3. سيتم عرض معلومات الوجه (اسم، ابتسامة، عيون)
4. سيتم حفظ صور الوجوه الجديدة
5. الضغط على زر 'q' لخروج من البرنامج

## الملفات
- `enhanced_face_detection.py`: الملف الرئيسي للبرنامج
- `requirements.txt`: قائمة المكتبات المطلوبة
- `face_captures/`: مجلد لحفظ صور الوجوه
- `face_stats/`: مجلد لحفظ الإحصائيات
- `known_faces.csv`: ملف لتخزين الوجوه المعروفة

## تنبيهات
- تأكد من وجود كاميرا تعمل على جهازك
- قد تحتاج إلى تثبيت مكتبات C++ Visual Studio Runtime
- قد تحتاج إلى تثبيت مكتبات OpenCV إضافية

## المساهمة
1. قم بعمل Fork للمشروع
2. قم بإنشاء فرع جديد (`git checkout -b feature/AmazingFeature`)
3. قم بعمل Commit للتغييرات (`git commit -m 'Add some AmazingFeature'`)
4. قم بدفع الفرع (`git push origin feature/AmazingFeature`)
5. قم بإنشاء Pull Request

## فريق الاعداد
1. يوسف جابر السيد دياب - 2220528
2. يوسف محمد يوسف مرعي - 2220538
3. محمد صالح محمد عبدالفتاح - 2220383

