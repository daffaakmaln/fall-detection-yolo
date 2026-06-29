import cv2
import os
import math
import pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")

def hitung_fitur(titik, conf_titik):
    """Hitung sudut kemiringan dan posisi dari titik-titik sendi."""
    c_bahu = min(conf_titik[5].item(), conf_titik[6].item())
    c_pinggul = min(conf_titik[11].item(), conf_titik[12].item())
    
    if c_bahu < 0.3 or c_pinggul < 0.3:
        return None
    
    bahu_x = (titik[5][0].item() + titik[6][0].item()) / 2
    bahu_y = (titik[5][1].item() + titik[6][1].item()) / 2
    pinggul_x = (titik[11][0].item() + titik[12][0].item()) / 2
    pinggul_y = (titik[11][1].item() + titik[12][1].item()) / 2
    
    delta_x = pinggul_x - bahu_x
    delta_y = pinggul_y - bahu_y
    sudut = math.degrees(math.atan2(abs(delta_x), abs(delta_y) + 0.0001))
    
    posisi_y = (bahu_y + pinggul_y) / 2
    
    return {
        "sudut": sudut,
        "posisi_y": posisi_y
    }

def proses_folder_gambar(folder_path, label, hasil_list):
    """Proses folder berisi gambar-gambar berurutan (seperti UR Fall dataset)."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
    
    posisi_sebelumnya = None
    
    for idx, fname in enumerate(files):
        img_path = os.path.join(folder_path, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        hasil = model(frame, verbose=False)
        keypoints = hasil[0].keypoints
        boxes = hasil[0].boxes
        
        best_idx = None
        best_conf = 0.3
        if boxes is not None:
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0 and float(box.conf[0]) > best_conf:
                    best_conf = float(box.conf[0])
                    best_idx = i
        
        if best_idx is not None and keypoints is not None and len(keypoints.xy) > best_idx:
            titik = keypoints.xy[best_idx]
            conf_titik = keypoints.conf[best_idx]
            
            if len(titik) >= 13:
                fitur = hitung_fitur(titik, conf_titik)
                
                if fitur is not None:
                    kecepatan = 0
                    if posisi_sebelumnya is not None:
                        kecepatan = (fitur["posisi_y"] - posisi_sebelumnya) / (1/30)
                    
                    # filter untuk fall label: only save kecepatan > 200 or sudut > 50
                    simpan = True
                    if label == "fall":
                        relevan = kecepatan > 200 or fitur["sudut"] > 50  
                        simpan = relevan
                    
                    if simpan:
                        hasil_list.append({
                            "sudut": fitur["sudut"],
                            "kecepatan": kecepatan,
                            "label": label
                        })
                    
                    posisi_sebelumnya = fitur["posisi_y"]
    
    print(f"  {folder_path}: {len(files)} frame diproses")

def proses_video(video_path, label, hasil_list):
    """Proses file video (untuk dataset custom)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    posisi_sebelumnya = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hasil = model(frame, verbose=False)
        keypoints = hasil[0].keypoints
        boxes = hasil[0].boxes
        
        best_idx = None
        best_conf = 0.3
        if boxes is not None:
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0 and float(box.conf[0]) > best_conf:
                    best_conf = float(box.conf[0])
                    best_idx = i
        
        if best_idx is not None and keypoints is not None and len(keypoints.xy) > best_idx:
            titik = keypoints.xy[best_idx]
            conf_titik = keypoints.conf[best_idx]
            
            if len(titik) >= 13:
                fitur = hitung_fitur(titik, conf_titik)
                
                if fitur is not None:
                    kecepatan = 0
                    if posisi_sebelumnya is not None:
                        kecepatan = (fitur["posisi_y"] - posisi_sebelumnya) / (1/fps)
                    
                    hasil_list.append({
                        "sudut": fitur["sudut"],
                        "kecepatan": kecepatan,
                        "label": label
                    })
                    
                    posisi_sebelumnya = fitur["posisi_y"]
        
        frame_count += 1
    
    cap.release()
    print(f"  {video_path}: {frame_count} frame diproses")

#main
hasil_list = []

#UR Fall - format gambar
print("Memproses dataset FALL...")
fall_path = "Fall"
for folder in os.listdir(fall_path):
    folder_path = os.path.join(fall_path, folder)
    if os.path.isdir(folder_path):
        proses_folder_gambar(folder_path, "fall", hasil_list)

#proses dataset Normal dari UR Fall (format gambar)
print("\nMemproses dataset NORMAL (ADL)...")
normal_path = "Normal"
for folder in os.listdir(normal_path):
    folder_path = os.path.join(normal_path, folder)
    if os.path.isdir(folder_path):
        proses_folder_gambar(folder_path, "normal", hasil_list)

#video custom (tiduran, duduk, dll)
print("\nMemproses dataset CUSTOM (video tiduran dll)...")
custom_path = "dataset_custom"
for subfolder in os.listdir(custom_path):
    subfolder_path = os.path.join(custom_path, subfolder)
    if os.path.isdir(subfolder_path):
        for video_file in os.listdir(subfolder_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(subfolder_path, video_file)
                proses_video(video_path, "normal", hasil_list)

df = pd.DataFrame(hasil_list)
df.to_csv("dataset_fitur.csv", index=False)

print(f"\n✅ SELESAI! Total {len(hasil_list)} baris data tersimpan ke dataset_fitur.csv")
print(f"   Fall: {len(df[df['label']=='fall'])} baris")
print(f"   Normal: {len(df[df['label']=='normal'])} baris")