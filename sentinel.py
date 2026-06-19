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

TELEGRAM_TOKEN    = "8944179544:AAEhNUAPgBuFqI922l7DlktjCBPaWF2VHqk"
TELEGRAM_CHAT_ID  = "5037364425"
JEDA_NOTIFIKASI   = 30
WAKTU_DIAM        = 1.5

# ─────────────────────────────────────────────────────────────
# FUNGSI TELEGRAM
# ─────────────────────────────────────────────────────────────

def cek_perintah():
    global status, waktu_jatuh
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
                    status = "AMAN"
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
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan}, timeout=5)
            if resp.status_code == 200:
                print(f"[Telegram] ✅ Pesan terkirim.")
            else:
                print(f"[Telegram] ⚠️ HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Telegram] ❌ Gagal kirim pesan: {e}")
    threading.Thread(target=_kirim, daemon=True).start()

def kirim_frame_ke_backend(frame):
    def _kirim():
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            url = f"{BACKEND_URL}/api/cameras/{CAMERA_ID}/status-frame"
            resp = requests.post(
                url,
                headers={"x-api-key": AI_API_KEY},
                files={"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=5
            )
            if resp.status_code == 200:
                print("[Backend] ✅ Frame terkirim.")
            else:
                print(f"[Backend] ⚠️ HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Backend] ❌ Gagal kirim frame: {e}")
    threading.Thread(target=_kirim, daemon=True).start()

def kirim_foto_telegram(frame):
    def _kirim():
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            resp = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": f"⚠️ *SentinelAI Alert*\nSeseorang terdeteksi JATUH!\n🕐 {time.strftime('%d/%m/%Y %H:%M:%S')}",
                    "parse_mode": "Markdown"
                },
                files={"photo": ("fall.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10
            )
            if resp.status_code == 200:
                print(f"[Telegram] ✅ Foto terkirim.")
            else:
                print(f"[Telegram] ⚠️ HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Telegram] ❌ Gagal kirim foto: {e}")
    threading.Thread(target=_kirim, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# FUNGSI HITUNG FITUR
# ─────────────────────────────────────────────────────────────

def hitung_fitur(titik, conf_titik):
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

    return {"sudut": sudut, "posisi_y": posisi_y}

# ─────────────────────────────────────────────────────────────
# INISIALISASI
# ─────────────────────────────────────────────────────────────

model_pose = YOLO("yolov8n-pose.pt")
model_rf   = joblib.load("model_fall_detection.pkl")
kamera     = cv2.VideoCapture(0)

waktu_jatuh                = None
status                     = "AMAN"
waktu_notifikasi_terakhir  = 0
frame_terkini              = None
posisi_sebelumnya          = None
waktu_sebelumnya           = None
waktu_kirim_frame_terakhir = 0

threading.Thread(target=cek_perintah, daemon=True).start()
print("SentinelAI (Hybrid RF) aktif! Tekan Q untuk keluar.")

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
    prediksi_label = "normal"
    prediksi_conf = 0

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

                if posisi_sebelumnya is not None and waktu_sebelumnya is not None:
                    dt = time.time() - waktu_sebelumnya
                    if dt > 0:
                        kecepatan = (fitur["posisi_y"] - posisi_sebelumnya) / dt

                # ── Prediksi pakai Random Forest ──────────────────
                X_input = np.array([[sudut, kecepatan]])
                prediksi_label = model_rf.predict(X_input)[0]
                prediksi_proba = model_rf.predict_proba(X_input)[0]
                prediksi_conf = max(prediksi_proba)

                # Jalur 1: prediksi RF yakin fall
                # Jalur 2: sudut sangat besar + kecepatan cukup tinggi (cadangan)
                jatuh_terdeteksi = (prediksi_label == "fall" and prediksi_conf > 0.6) or (sudut > 70 and kecepatan > 300)

                # ── DEBUG PRINT ────────────────────────────────────
                print(f"Sudut: {sudut:.1f}, Kecepatan: {kecepatan:.1f}, RF: {prediksi_label} ({prediksi_conf:.2f}), Jatuh: {jatuh_terdeteksi}")

                posisi_sebelumnya = fitur["posisi_y"]
                waktu_sebelumnya = time.time()
    else:
        posisi_sebelumnya = None
        waktu_sebelumnya = None

    # ── Logika status + notifikasi Telegram ──────────────────
    if jatuh_terdeteksi:
        if waktu_jatuh is None:
            waktu_jatuh = time.time()
        elif time.time() - waktu_jatuh >= WAKTU_DIAM:
            if status != "JATUH TERDETEKSI!":
                status   = "JATUH TERDETEKSI!"
                sekarang = time.time()
                if sekarang - waktu_notifikasi_terakhir > JEDA_NOTIFIKASI:
                    waktu_notifikasi_terakhir = sekarang
                    frame_notif = hasil[0].plot()
                    kirim_telegram(
                        f"🚨 PERINGATAN SentinelAI!\n"
                        f"Seseorang terdeteksi JATUH!\n"
                        f"🕐 Waktu: {time.strftime('%H:%M:%S')}\n"
                        f"📷 Screenshot terlampir."
                    )
                    kirim_foto_telegram(frame_notif)
    else:
        # Hanya reset kalau status BELUM "JATUH TERDETEKSI!"
        if status != "JATUH TERDETEKSI!":
            waktu_jatuh = None
            status      = "AMAN"

    # ── Tampilan overlay ──────────────────────────────────────
    warna = (0, 0, 255) if "JATUH" in status else (0, 255, 0)
    frame_annotated = hasil[0].plot()
    frame_terkini   = frame_annotated.copy()

    # ── Kirim frame terbaru ke backend secara berkala ─────────
    sekarang_frame = time.time()
    if sekarang_frame - waktu_kirim_frame_terakhir >= JEDA_KIRIM_FRAME:
        waktu_kirim_frame_terakhir = sekarang_frame
        kirim_frame_ke_backend(frame_annotated)

    cv2.putText(frame_annotated, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)
    cv2.putText(frame_annotated, f"Sudut: {sudut:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Kecepatan: {kecepatan:.1f}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Prediksi RF: {prediksi_label} ({prediksi_conf:.1%})", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("SentinelAI - Hybrid Fall Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
#test