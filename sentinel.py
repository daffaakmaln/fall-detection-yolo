import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8n-pose.pt")
kamera = cv2.VideoCapture(0)

KECEPATAN_JATUH = 80
RASIO_REBAH = 1.3
WAKTU_DIAM = 2

waktu_jatuh = None
status = "AMAN"
posisi_sebelumnya = None
waktu_sebelumnya = None
riwayat_kecepatan = []

print("GuardianEye aktif! Tekan Q untuk keluar.")

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil = model(frame, verbose=False)
    jatuh_terdeteksi = False
    posisi_sekarang = None
    kecepatan_turun = 0
    rasio = 0

    boxes = hasil[0].boxes

    # Ambil person dengan confidence tertinggi (bukan skip kalau < 0.6)
    best_box = None
    best_conf = 0.3  # minimum confidence diturunkan ke 0.3
    for box in (boxes or []):
        if int(box.cls[0]) == 0:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0]
        lebar = (x2 - x1).item()
        tinggi = (y2 - y1).item()
        rasio = lebar / tinggi if tinggi > 0 else 0
        posisi_sekarang = y1.item()

        if posisi_sebelumnya is not None and waktu_sebelumnya is not None:
            dt = time.time() - waktu_sebelumnya
            if dt > 0:
                kecepatan_turun = (posisi_sekarang - posisi_sebelumnya) / dt

        # Filter noise: abaikan lonjakan ekstrem
        if -300 < kecepatan_turun < 300:
            riwayat_kecepatan.append(kecepatan_turun)
        if len(riwayat_kecepatan) > 10:
            riwayat_kecepatan.pop(0)

        rata_kecepatan = sum(riwayat_kecepatan) / len(riwayat_kecepatan) if riwayat_kecepatan else 0

        sedang_rebah = rasio > RASIO_REBAH

        if sedang_rebah and rata_kecepatan > KECEPATAN_JATUH:
            jatuh_terdeteksi = True

        # Reset riwayat kalau sudah berdiri
        if rasio < 0.8:
            riwayat_kecepatan = []

    if posisi_sekarang is not None:
        posisi_sebelumnya = posisi_sekarang
        waktu_sebelumnya = time.time()
    else:
        riwayat_kecepatan = []
        waktu_jatuh = None
        status = "AMAN"

    if jatuh_terdeteksi:
        if waktu_jatuh is None:
            waktu_jatuh = time.time()
        elif time.time() - waktu_jatuh >= WAKTU_DIAM:
            status = "JATUH TERDETEKSI!"
    else:
        waktu_jatuh = None
        status = "AMAN"

    rata_display = sum(riwayat_kecepatan) / len(riwayat_kecepatan) if riwayat_kecepatan else 0
    warna = (0, 0, 255) if "JATUH" in status else (0, 255, 0)
    frame_annotated = hasil[0].plot()
    cv2.putText(frame_annotated, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)
    cv2.putText(frame_annotated, f"Kecepatan: {kecepatan_turun:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Rata kecepatan: {rata_display:.1f}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Rasio: {rasio:.2f}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("GuardianEye - Fall Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()