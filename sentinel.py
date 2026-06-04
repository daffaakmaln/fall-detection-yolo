import cv2
from ultralytics import YOLO
import time
import requests
import threading

# ╔══════════════════════════════════════════════════════════════╗
# ║              KONFIGURASI TELEGRAM BOT                       ║
# ║  1. Buka Telegram → cari @BotFather → /newbot               ║
# ║  2. Copy token yang diberikan ke TELEGRAM_TOKEN             ║
# ║  3. Cari @userinfobot → kirim pesan → copy ID ke CHAT_ID    ║
# ╚══════════════════════════════════════════════════════════════╝
TELEGRAM_TOKEN   = "8944179544:AAEhNUAPgBuFqI922l7DlktjCBPaWF2VHqk"       # contoh: "7123456789:AAFxxxxxxx"
TELEGRAM_CHAT_ID = "5037364425"         # contoh: "123456789"
JEDA_NOTIFIKASI  = 30                          # detik antar notifikasi (cegah spam)

# ── KONFIGURASI DETEKSI ───────────────────────────────────────
KECEPATAN_JATUH = 45
RASIO_REBAH     = 1.3
WAKTU_DIAM      = 2


# ─────────────────────────────────────────────────────────────
# FUNGSI TELEGRAM
# ─────────────────────────────────────────────────────────────

def cek_perintah():
    """Polling perintah dari Telegram secara terus-menerus."""
    global status, waktu_jatuh, riwayat_kecepatan
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
                    continue  # abaikan pesan dari orang lain

                if pesan == "/status":
                    kirim_telegram(f"📊 Status saat ini: {status}")
                    if frame_terkini is not None:
                        kirim_foto_telegram(frame_terkini)  # kirim foto keadaan sekarang

                elif pesan == "/start":
                    kirim_telegram(
                        "👋 Selamat datang di SentinelAI!\n"
                        "Saya akan memantau dan mendeteksi jika seseorang jatuh.\n"
                        "Kirim /help untuk melihat perintah yang tersedia."
                    )
                elif pesan == "/reset":
                    waktu_jatuh = None
                    riwayat_kecepatan = []
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
    """Kirim pesan teks ke Telegram di background thread."""
    def _kirim():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan},
                timeout=5
            )
            if resp.status_code == 200:
                print(f"[Telegram] ✅ Pesan terkirim.")
            else:
                print(f"[Telegram] ⚠️  HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Telegram] ❌ Gagal kirim pesan: {e}")
    threading.Thread(target=_kirim, daemon=True).start()


def kirim_foto_telegram(frame):
    """Kirim screenshot frame ke Telegram di background thread."""
    def _kirim():
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            resp = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": (
                        f"⚠️ *SentinelAI Alert*\n"
                        f"Keadaan sekarang!\n"
                        f"🕐 {time.strftime('%d/%m/%Y %H:%M:%S')}"
                    ),
                    "parse_mode": "Markdown"
                },
                files={"photo": ("fall.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10
            )
            if resp.status_code == 200:
                print(f"[Telegram] ✅ Foto terkirim.")
            else:
                print(f"[Telegram] ⚠️  HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Telegram] ❌ Gagal kirim foto: {e}")
    threading.Thread(target=_kirim, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# INISIALISASI
# ─────────────────────────────────────────────────────────────

model  = YOLO("yolov8n-pose.pt")
kamera = cv2.VideoCapture(0)

waktu_jatuh              = None
status                   = "AMAN"
posisi_sebelumnya        = None
waktu_sebelumnya         = None
riwayat_kecepatan        = []
waktu_notifikasi_terakhir = 0
frame_terkini = None

threading.Thread(target=cek_perintah, daemon=True).start()
print("SentinelAI aktif! Tekan Q untuk keluar.")

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil           = model(frame, verbose=False)
    jatuh_terdeteksi = False
    posisi_sekarang  = None
    kecepatan_turun  = 0
    rasio            = 0

    boxes = hasil[0].boxes

    for i, box in enumerate(boxes or []):
        if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.6:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        lebar  = (x2 - x1).item()
        tinggi = (y2 - y1).item()
        rasio  = lebar / tinggi if tinggi > 0 else 0
        posisi_sekarang = y1.item()

        if posisi_sebelumnya is not None and waktu_sebelumnya is not None:
            dt = time.time() - waktu_sebelumnya
            if dt > 0:
                kecepatan_turun = (posisi_sekarang - posisi_sebelumnya) / dt

        # Simpan riwayat kecepatan (max 15 frame ≈ 0.5 detik)
        riwayat_kecepatan.append(kecepatan_turun)
        if len(riwayat_kecepatan) > 15:
            riwayat_kecepatan.pop(0)

        kecepatan_max = max(riwayat_kecepatan)
        sedang_rebah  = rasio > RASIO_REBAH

        # Jatuh = rebah SEKARANG + kecepatan tinggi dalam waktu dekat
        if sedang_rebah and kecepatan_max > KECEPATAN_JATUH:
            jatuh_terdeteksi = True
            break

        # Reset riwayat kalau sudah berdiri tegak
        if rasio < 0.8:
            riwayat_kecepatan = []

    if posisi_sekarang is not None:
        posisi_sebelumnya = posisi_sekarang
        waktu_sebelumnya  = time.time()
    else:
        riwayat_kecepatan = []
        waktu_jatuh       = None
        status            = "AMAN"

    # ── Logika status + notifikasi Telegram ──────────────────
    if jatuh_terdeteksi:
        if waktu_jatuh is None:
            waktu_jatuh = time.time()
        elif time.time() - waktu_jatuh >= WAKTU_DIAM:
            if status != "JATUH TERDETEKSI!":
                # Status baru berubah → kirim notifikasi
                status   = "JATUH TERDETEKSI!"
                sekarang = time.time()
                if sekarang - waktu_notifikasi_terakhir > JEDA_NOTIFIKASI:
                    waktu_notifikasi_terakhir = sekarang
                    frame_notif = hasil[0].plot()   # frame dengan anotasi YOLO
                    kirim_telegram(
                        f"🚨 PERINGATAN SentinelEye!\n"
                        f"Seseorang terdeteksi JATUH!\n"
                        f"🕐 Waktu: {time.strftime('%H:%M:%S')}\n"
                        f"📷 Screenshot terlampir."
                    )
                    kirim_foto_telegram(frame_notif)
            else:
                status = "JATUH TERDETEKSI!"
    else:
        waktu_jatuh = None
        status      = "AMAN"
    # ─────────────────────────────────────────────────────────

    # ── Tampilan overlay ─────────────────────────────────────
    kecepatan_max_display = max(riwayat_kecepatan) if riwayat_kecepatan else 0
    warna          = (0, 0, 255) if "JATUH" in status else (0, 255, 0)
    frame_annotated = hasil[0].plot()
    frame_terkini = frame_annotated.copy()  # ← tambahkan ini

    cv2.putText(frame_annotated, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)
    cv2.putText(frame_annotated, f"Kecepatan: {kecepatan_turun:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Kecepatan max: {kecepatan_max_display:.1f}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame_annotated, f"Rasio: {rasio:.2f}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("SentinelEye - Fall Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()