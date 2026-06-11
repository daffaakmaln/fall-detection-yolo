import cv2
from ultralytics import YOLO
import time
import requests
import threading

BACKEND_URL    = "http://localhost:3000"
AI_API_KEY     = "kunci_rahasia_gacor_jovanvendaf"  # samakan dengan .env backend
CAMERA_ID      = 1  # ID kamera ini di database (sesuaikan dengan ID kamera kamu)
JEDA_KIRIM_FRAME = 5  # kirim foto terbaru setiap 5 detik
TELEGRAM_TOKEN   = "8944179544:AAEhNUAPgBuFqI922l7DlktjCBPaWF2VHqk"
TELEGRAM_CHAT_ID = "5037364425"
JEDA_NOTIFIKASI  = 30
WAKTU_DIAM       = 2

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

# ── INISIALISASI ──────────────────────────────────────────────
model  = YOLO("best.pt")
kamera = cv2.VideoCapture(0)

waktu_jatuh               = None
status                    = "AMAN"
waktu_notifikasi_terakhir = 0
frame_terkini             = None
waktu_kirim_frame_terakhir = 0

threading.Thread(target=cek_perintah, daemon=True).start()
print("SentinelAI aktif! Tekan Q untuk keluar.")

# ── MAIN LOOP ─────────────────────────────────────────────────
while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil = model(frame, verbose=False)

    # Klasifikasi pakai model fine-tuned
    label      = hasil[0].names[hasil[0].probs.top1]
    confidence = hasil[0].probs.top1conf.item()

    jatuh_terdeteksi = label == "fall" and confidence > 0.7

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
                    kirim_telegram(
                        f"🚨 PERINGATAN SentinelAI!\n"
                        f"Seseorang terdeteksi JATUH!\n"
                        f"🕐 Waktu: {time.strftime('%H:%M:%S')}\n"
                        f"📷 Screenshot terlampir."
                    )
                    kirim_foto_telegram(frame)
    else:
        waktu_jatuh = None
        status      = "AMAN"

    # ── Tampilan overlay ──────────────────────────────────────
    warna          = (0, 0, 255) if "JATUH" in status else (0, 255, 0)
    frame_annotated = frame.copy()
    frame_terkini   = frame_annotated.copy()
    # ── Kirim frame terbaru ke backend secara berkala ─────────
    sekarang_frame = time.time()
    if sekarang_frame - waktu_kirim_frame_terakhir >= JEDA_KIRIM_FRAME:
        waktu_kirim_frame_terakhir = sekarang_frame
        kirim_frame_ke_backend(frame_annotated)

    cv2.putText(frame_annotated, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)
    cv2.putText(frame_annotated, f"{label}: {confidence:.2%}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("SentinelAI - Fall Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()