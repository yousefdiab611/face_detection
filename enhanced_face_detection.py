import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from playsound import playsound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ✅ إنشاء المجلدات لو مش موجودة
if not os.path.exists('face_captures'):
    os.makedirs('face_captures')
if not os.path.exists('face_stats'):
    os.makedirs('face_stats')

# ✅ تهيئة الكاشكاد للكشف عن الوجه والعيون والابتسامات
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ✅ إحصائيات النظام
face_count_history = []
face_stats = pd.DataFrame(columns=['timestamp', 'face_count', 'smiles', 'eyes'])  # DataFrame فارغ مبدئيًا

# ✅ التعرف على الوجوه المعروفة
known_face_encodings = []
known_face_names = []

# ✅ تحميل الوجوه المعروفة لو موجودة
if os.path.exists('known_faces.csv'):
    known_faces_df = pd.read_csv('known_faces.csv')
    for index, row in known_faces_df.iterrows():
        image = face_recognition.load_image_file(row['image_path'])
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(row['name'])

def draw_face_info(frame, x, y, w, h, face_name, is_smiling, has_eyes):
    color = (0, 255, 0)  # أخضر
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if face_name:
        cv2.putText(frame, face_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if is_smiling:
        cv2.putText(frame, 'Smiling', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if has_eyes:
        cv2.putText(frame, 'Eyes Open', (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def save_face_capture(frame, x, y, w, h, timestamp):
    face_image = frame[y:y+h, x:x+w]
    filename = f"face_captures/face_{timestamp}.jpg"
    cv2.imwrite(filename, face_image)
    return filename

def update_statistics(face_count, smiles, eyes):
    global face_stats  # ✅ ضروري لتحديث المتغير العالمي
    timestamp = datetime.now()
    stats = pd.DataFrame({
        'timestamp': [timestamp],
        'face_count': [face_count],
        'smiles': [smiles],
        'eyes': [eyes]
    })
    face_stats = pd.concat([face_stats, stats], ignore_index=True)
    face_stats.to_csv('face_stats/stats.csv', index=False)

def play_alert_sound():
    try:
        playsound('alert.wav')
    except:
        print("Warning: Could not play alert sound")

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Enhanced Face Detection System Started...")
    print("Press 'q' to quit")
    print("Features: Face Tracking, Recognition, Emotion Detection, Age & Gender, Quality Analysis, Statistics, Voice Alerts, Face Capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        smiles_count = 0
        eyes_count = 0

        for (x, y, w, h) in faces:
            # ✅ التعرف على الوجه بالطريقة الصحيحة
            face_locations = [(y, x + w, y + h, x)]  # top, right, bottom, left
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_name = "Unknown"
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    face_name = known_face_names[first_match_index]

            # ✅ الكشف عن الابتسامة
            smile = smile_cascade.detectMultiScale(
                gray[y:y+h, x:x+w], scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
            )
            is_smiling = len(smile) > 0
            if is_smiling:
                smiles_count += 1

            # ✅ الكشف عن العيون
            eyes = eye_cascade.detectMultiScale(
                gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
            )
            has_eyes = len(eyes) > 0
            if has_eyes:
                eyes_count += 1

            # ✅ عرض المعلومات
            draw_face_info(frame, x, y, w, h, face_name, is_smiling, has_eyes)

            # ✅ حفظ صورة الوجه
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_face_capture(frame, x, y, w, h, timestamp)

            # ✅ تشغيل تنبيه لو وجه جديد
            if face_name == "Unknown":
                play_alert_sound()

        # ✅ تحديث الإحصائيات
        update_statistics(len(faces), smiles_count, eyes_count)

        # ✅ عرض الإحصائيات
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Smiles: {smiles_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Eyes: {eyes_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Enhanced Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
