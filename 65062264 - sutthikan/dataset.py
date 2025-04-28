import cv2
import dlib
import os

# กำหนด path วิดีโอ
video_path = r"C:\CPE\PythonOpencvtest\video\IMG_9279.MOV"  # ระวัง path ใส่ r"" ไว้ด้วยนะ
# กำหนดโฟลเดอร์เก็บใบหน้าที่ตรวจเจอ
faces_folder = "extracted_faces"
os.makedirs(faces_folder, exist_ok=True)

# โหลด detector ของ dlib
detector = dlib.get_frontal_face_detector()

# เปิดวิดีโอ
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดไฟล์วิดีโอได้!")
    exit()

frame_index = 0
face_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # จบวิดีโอ

    # แปลงเป็น grayscale (dlib ทำงานกับ grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = detector(gray)

    for face in faces:
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # ตัดเฉพาะใบหน้าออกมา
        face_crop = frame[y:y2, x:x2]

        if face_crop.size == 0:
            continue

        # เซฟใบหน้าเป็นไฟล์รูป
        face_filename = os.path.join(faces_folder, f"face_{face_index:04d}.jpg")
        cv2.imwrite(face_filename, face_crop)
        print(f"✅ บันทึกใบหน้า {face_filename}")
        face_index += 1

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ ดึงใบหน้าทั้งหมด {face_index} รูป เก็บในโฟลเดอร์ '{faces_folder}' สำเร็จแล้ว!")
