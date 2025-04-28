import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import json
import requests
from datetime import datetime, timedelta, timezone
import time

# -------------------- โหลดโมเดลและฐานข้อมูล --------------------
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=60, min_face_size=20)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

face_database_path = "face_database.npy"
if os.path.exists(face_database_path):
    data = np.load(face_database_path, allow_pickle=True).item()
    if isinstance(data, dict):
        face_embeddings = data
    else:
        print("❌ ข้อมูลใน face_database.npy ไม่ใช่ dictionary!")
        exit()
else:
    print("❌ ไม่พบไฟล์ face_database.npy!")
    exit()

# -------------------- ฟังก์ชัน --------------------
def recognize_face_with_mtcnn(face_resized):
    try:
        if face_resized is None or face_resized.size == 0:
            return None
        face_resized = cv2.resize(face_resized, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            embedding = facenet_model(face_tensor)
        return embedding.squeeze().numpy()
    except Exception as e:
        print("⚠️ Error in face embedding:", e)
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------- การตั้งค่า --------------------
server_url = "https://d076-49-237-33-240.ngrok-free.app/api/student-checks"
output_folder = "recorded_videos"
detected_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(detected_folder, exist_ok=True)

clip_duration = 2 * 60  # 2 นาที
fps = 15
THAI_TIMEZONE = timezone(timedelta(hours=7))

# -------------------- กล้อง RTSP --------------------
camera_ip = 'rtsp://Face495:cpe495@172.20.10.3:554/stream1'
capture = cv2.VideoCapture(camera_ip, cv2.CAP_FFMPEG)

if not capture.isOpened():
    print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ FFmpeg")
    capture = cv2.VideoCapture(camera_ip, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ GStreamer")
        exit()
print("✅ เชื่อมต่อกับกล้อง RTSP สำเร็จ!")

capture.set(3, 1280)
capture.set(4, 720)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# -------------------- Loop หลัก --------------------
while True:
    # 1. บันทึกวิดีโอ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_filename = os.path.join(output_folder, f"record_{timestamp}.mp4")
    done_filename = video_filename + ".done"
    video_out = cv2.VideoWriter(video_filename, fourcc, fps, (1280, 720))

    print(f"📹 เริ่มบันทึกวิดีโอ: {video_filename}")
    start_time = time.time()

    while time.time() - start_time < clip_duration:
        ret, frame = capture.read()
        if not ret:
            print("❌ Error: ไม่สามารถอ่านเฟรมจากกล้อง!")
            break
        video_out.write(frame)
        cv2.imshow('Recording (Face Detection)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            exit()

    video_out.release()
    with open(done_filename, "w") as f:
        f.write("done")
    print(f"✅ บันทึกเสร็จสิ้น: {video_filename}")

    # 2. ประมวลผลวิดีโอที่เพิ่งบันทึก
    print(f"📂 กำลังประมวลผลวิดีโอ: {video_filename}")
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดไฟล์วิดีโอได้!")
        continue

    frame_index = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_faces = set()

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                if (x1, y1, x2, y2) not in saved_faces:
                    face_filename = os.path.join(detected_folder, f"face_{frame_count:04d}.jpg")
                    cv2.imwrite(face_filename, face)
                    saved_faces.add((x1, y1, x2, y2))
                    frame_count += 1

        frame_index += 10

    cap.release()
    cv2.destroyAllWindows()
    print("✅ การตรวจจับใบหน้าเสร็จสิ้น!")

    # 3. เปรียบเทียบใบหน้า
    print("🔍 กำลังเปรียบเทียบใบหน้ากับฐานข้อมูล...")
    existing_names = set()

    for face_file in os.listdir(detected_folder):
        face_path = os.path.join(detected_folder, face_file)
        face_img = cv2.imread(face_path)
        face_embedding = recognize_face_with_mtcnn(face_img)

        if face_embedding is not None:
            name = "Unknown"
            best_similarity = float("-inf")

            for db_name, db_embedding in face_embeddings.items():
                similarity = cosine_similarity(face_embedding, db_embedding)
                print(f"🔍 เปรียบเทียบ {face_file} กับ {db_name} | ค่าความคล้าย: {similarity:.2f}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    name = db_name

            if best_similarity >= 0.75 and name not in existing_names:
                print(f"📌 ใบหน้าใน {face_file} ตรงกับ {name} (ค่าความคล้าย {best_similarity:.2f})")
                now_th = datetime.now(THAI_TIMEZONE)

                attendance_data = {
                    "Student_ID": int(name),
                    "Course_ID": "CPE451",
                    "Check_Date": now_th.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                    "Check_Time": now_th.strftime("%H:%M:%S")
                }

                try:
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(server_url, json=attendance_data, headers=headers)
                    if response.status_code == 200:
                        print(f"✅ ส่งข้อมูลสำเร็จ: {name}")
                    else:
                        print(f"❌ ส่งข้อมูลล้มเหลว: {name} | Status Code: {response.status_code} | Response: {response.text}")
                except Exception as e:
                    print(f"⚠️ Error sending data: {e}")

                existing_names.add(name)

        os.remove(face_path)
        print(f"❌ ลบไฟล์: {face_file}")

    os.remove(done_filename)
    print(f"✅ ลบไฟล์ .done สำหรับ {video_filename}")
    print("🔄 รอคลิปถัดไป...")

