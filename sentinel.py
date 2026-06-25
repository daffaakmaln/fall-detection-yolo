import cv2
from ultralytics import YOLO
import time
import math
import requests
import threading
import joblib
import numpy as np

# ╔══════════════════════════════════════════════════════════════╗
# ║              KONFIGURASI                                     ║
# ╚══════════════════════════════════════════════════════════════╝
BACKEND_URL       = "http://localhost:3000"
AI_API_KEY        = "kunci_rahasia_gacor_jovanvendaf"
CAMERA_ID         = 2
JEDA_KIRIM_FRAME  = 5

TELEGRAM_TOKEN    = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID  = "YOUR_CHAT_ID"
JEDA_NOTIFIKASI   = 30
WAKTU_DIAM        = 1.5

# Threshold hybrid deteksi jatuh
RF_FALL_CONF_THRESHOLD      = 0.60
IMPACT_SPEED_THRESHOLD      = 95
IMPACT_DELTA_ANGLE          = 10
IMPACT_ANGLE_THRESHOLD      = 40
IMPACT_WINDOW_SEC           = 0.6
IMPACT_VOTE_WINDOW          = 5
IMPACT_VOTE_REQUIRED        = 2
KECEPATAN_PUNCAK_WINDOW     = 30
POST_FALL_CONFIRM_WINDOW    = 2.0
POST_FALL_REQUIRED_FRAMES   = 6
POST_FALL_ANGLE_THRESHOLD   = 65
POST_FALL_Y_OFFSET          = 60
RECOVER_ANGLE_THRESHOLD     = 35
RECOVER_TIME_SEC            = 2.0

# Tambahan smoothing
EMA_ALPHA = 0.7

# ─────────────────────────────────────────────────────────────
# FUNGSI TELEGRAM
# ─────────────────────────────────────────────────────────────

def cek_perintah():
    global status, waktu_jatuh, fall_state, suspect_start_time, recover_start_time, baseline_posisi_y
    offset = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            resp = requests.get(url, params={"offset": offset, "timeout": 5}, timeout=10)
            updates = resp.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                pesan = update.get("message", {}).get("text", "")
                chat_id = update.get("message", {}).get("chat", {}).get("id", "")

                if str(chat_id) != str(TELEGRAM_CHAT_ID):
                    continue

                if pesan == "/status":
                    kirim_telegram(f"📊 Status saat ini: {status}")
                    if frame_terkini is not None:
                        kirim_foto_telegram(frame_terkini)

                elif pesan == "/start":
                    kirim_telegram(
                        "👋 Selamat datang di SentinelAI!\n"
                        "Saya akan memantau dan mendeteksi jika seseorang jatuh.\n"
                        "Kirim /help untuk melihat perintah yang tersedia."
                    )

                elif pesan == "/reset":
                    waktu_jatuh = None
                    fall_state = "SAFE"
                    suspect_start_time = None
                    recover_start_time = None
                    baseline_posisi_y = None
                    status = "AMAN"

                    riwayat_prediksi.clear()
                    riwayat_kecepatan_max.clear()
                    riwayat_post_fall.clear()
                    riwayat_kecepatan.clear()

                    kirim_telegram("🔄 Sistem berhasil di-reset.")

                elif pesan == "/help":
                    kirim_telegram(
                        "📋 *Daftar Perintah Sentinel_AI:*\n"
                        "/status — Cek status deteksi saat ini\n"
                        "/reset  — Reset sistem ke AMAN\n"
                        "/help   — Tampilkan daftar perintah"
                    )

        except Exception as e:
            print(f"[Polling] Error: {e}")

        time.sleep(2)


def kirim_telegram(pesan):
    def _kirim():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan}, timeout=5)
        except Exception as e:
            print(f"[Telegram] ❌ {e}")
    threading.Thread(target=_kirim, daemon=True).start()


def kirim_foto_telegram(frame):
    def _kirim():
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": "⚠️ SentinelAI Alert"
                },
                files={"photo": ("fall.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10
            )
        except Exception as e:
            print(f"[Telegram] ❌ {e}")
    threading.Thread(target=_kirim, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# FUNGSI FITUR
# ─────────────────────────────────────────────────────────────

def hitung_fitur(titik, conf_titik):
    c_bahu = min(conf_titik[5].item(), conf_titik[6].item())
    c_pinggul = min(conf_titik[11].item(), conf_titik[12].item())

    # confidence gating lebih ketat
    if c_bahu < 0.5 or c_pinggul < 0.5:
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


# ─────────────────────────────────────────────────────────────
# INISIALISASI
# ─────────────────────────────────────────────────────────────

model_pose = YOLO("yolov8s-pose.pt")  # bisa ganti ke n/s/m
model_rf = joblib.load("model_fall_detection.pkl")
kamera = cv2.VideoCapture(0)

waktu_jatuh                = None
status                     = "AMAN"
waktu_notifikasi_terakhir  = 0
frame_terkini              = None
posisi_sebelumnya          = None
posisi_smooth              = None
waktu_sebelumnya           = None
waktu_kirim_frame_terakhir = 0

riwayat_prediksi           = []
riwayat_kecepatan_max      = []
riwayat_kecepatan          = []
riwayat_post_fall          = []

fall_state                 = "SAFE"
suspect_start_time         = None
recover_start_time         = None
baseline_posisi_y          = None
sudut_sebelumnya           = None

threading.Thread(target=cek_perintah, daemon=True).start()

print("SentinelAI aktif! Tekan Q untuk keluar.")

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil = model_pose(frame, verbose=False)
    keypoints = hasil[0].keypoints
    boxes = hasil[0].boxes

    jatuh_terdeteksi = False
    sudut = 0
    kecepatan = 0
    kecepatan_filtered = 0
    prediksi_label = "normal"
    prediksi_conf = 0
    kecepatan_puncak = 0
    delta_sudut = 0
    post_fall_condition = False

    sekarang = time.time()

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
                sudut = fitur["sudut"]

                # EMA smoothing
                if posisi_smooth is None:
                    posisi_smooth = fitur["posisi_y"]
                else:
                    posisi_smooth = (
                        EMA_ALPHA * posisi_smooth +
                        (1 - EMA_ALPHA) * fitur["posisi_y"]
                    )

                if posisi_sebelumnya is not None and waktu_sebelumnya is not None:
                    dt = sekarang - waktu_sebelumnya

                    if dt > 0:
                        kecepatan = (posisi_smooth - posisi_sebelumnya) / dt

                        # median filter
                        riwayat_kecepatan.append(kecepatan)
                        if len(riwayat_kecepatan) > 5:
                            riwayat_kecepatan.pop(0)

                        kecepatan_filtered = np.median(riwayat_kecepatan)

                        if dt <= IMPACT_WINDOW_SEC and sudut_sebelumnya is not None:
                            delta_sudut = abs(sudut - sudut_sebelumnya)

                # Random Forest
                X_input = np.array([[sudut, kecepatan_filtered]])
                prediksi_label = model_rf.predict(X_input)[0]
                prediksi_proba = model_rf.predict_proba(X_input)[0]
                prediksi_conf = max(prediksi_proba)

                # peak speed
                riwayat_kecepatan_max.append(kecepatan_filtered)
                if len(riwayat_kecepatan_max) > KECEPATAN_PUNCAK_WINDOW:
                    riwayat_kecepatan_max.pop(0)

                kecepatan_puncak = max(riwayat_kecepatan_max)

                prediksi_binary = 1 if (
                    prediksi_label == "fall" and
                    prediksi_conf > RF_FALL_CONF_THRESHOLD
                ) else 0

                riwayat_prediksi.append(prediksi_binary)
                if len(riwayat_prediksi) > IMPACT_VOTE_WINDOW:
                    riwayat_prediksi.pop(0)

                impact_vote_ok = sum(riwayat_prediksi) >= IMPACT_VOTE_REQUIRED

                impact_terdeteksi = (
                    kecepatan_puncak > IMPACT_SPEED_THRESHOLD and
                    sudut > IMPACT_ANGLE_THRESHOLD and
                    impact_vote_ok
                )

                # baseline aman
                if fall_state == "SAFE" and sudut < 45:
                    if baseline_posisi_y is None:
                        baseline_posisi_y = posisi_smooth
                    else:
                        baseline_posisi_y = 0.9 * baseline_posisi_y + 0.1 * posisi_smooth

                post_fall_condition = (
                    sudut > POST_FALL_ANGLE_THRESHOLD or
                    (
                        baseline_posisi_y is not None and
                        posisi_smooth > (baseline_posisi_y + POST_FALL_Y_OFFSET)
                    )
                )

                if fall_state == "SAFE":
                    if impact_terdeteksi:
                        fall_state = "SUSPECT_FALL"
                        suspect_start_time = sekarang
                        recover_start_time = None
                        riwayat_post_fall = []

                elif fall_state == "SUSPECT_FALL":
                    riwayat_post_fall.append(1 if post_fall_condition else 0)

                    if len(riwayat_post_fall) > 30:
                        riwayat_post_fall.pop(0)

                    confirm_hits = sum(riwayat_post_fall)

                    if confirm_hits >= POST_FALL_REQUIRED_FRAMES:
                        fall_state = "FALL_CONFIRMED"
                        status = "JATUH TERDETEKSI!"

                elif fall_state == "FALL_CONFIRMED":
                    if sudut < RECOVER_ANGLE_THRESHOLD:
                        if recover_start_time is None:
                            recover_start_time = sekarang
                        elif (sekarang - recover_start_time) >= RECOVER_TIME_SEC:
                            fall_state = "SAFE"
                            status = "AMAN"
                            recover_start_time = None
                            riwayat_post_fall = []
                            riwayat_prediksi = []
                            riwayat_kecepatan_max = []
                            riwayat_kecepatan = []

                if sudut < 30:
                    riwayat_prediksi = []
                    riwayat_kecepatan_max = []
                    riwayat_kecepatan = []

                jatuh_terdeteksi = (fall_state == "FALL_CONFIRMED")

                posisi_sebelumnya = posisi_smooth
                waktu_sebelumnya = sekarang
                sudut_sebelumnya = sudut

    else:
        posisi_sebelumnya = None
        waktu_sebelumnya = None
        sudut_sebelumnya = None

    if fall_state == "SAFE":
        status = "AMAN"
    elif fall_state == "SUSPECT_FALL":
        status = "TERDUGA JATUH"
    elif fall_state == "FALL_CONFIRMED":
        status = "JATUH TERDETEKSI!"

    # ── Overlay lengkap seperti code awal ───────────────────
    warna = (0, 0, 255) if "JATUH" in status else (0, 255, 0)

    frame_annotated = hasil[0].plot()
    frame_terkini = frame_annotated.copy()

    cv2.putText(frame_annotated, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)

    cv2.putText(frame_annotated, f"Sudut: {sudut:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"Kecepatan: {kecepatan:.1f}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"Kecepatan Filtered: {kecepatan_filtered:.1f}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"Puncak: {kecepatan_puncak:.1f}", (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"RF: {prediksi_label} ({prediksi_conf:.1%})", (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"Impact Vote: {sum(riwayat_prediksi)}/{IMPACT_VOTE_WINDOW}", (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame_annotated, f"Post-fall: {sum(riwayat_post_fall)}/{POST_FALL_REQUIRED_FRAMES}", (30, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("SentinelAI - Hybrid Fall Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()