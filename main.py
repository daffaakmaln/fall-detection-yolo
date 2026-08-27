import os
import sys
import time
import math
import threading

import cv2
import numpy as np
import joblib
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFrame, QMessageBox
)

load_dotenv()

# ============================================================
#  KONFIGURASI (identik dengan script asli)
# ============================================================
BACKEND_URL       = "http://localhost:3000"
AI_API_KEY        = os.getenv("API_KEY")
JEDA_KIRIM_FRAME   = 5

RF_FALL_CONF_THRESHOLD    = 0.60
IMPACT_SPEED_THRESHOLD    = 95
IMPACT_ANGLE_THRESHOLD    = 40
IMPACT_WINDOW_SEC         = 0.6
IMPACT_VOTE_WINDOW        = 5
IMPACT_VOTE_REQUIRED      = 2
KECEPATAN_PUNCAK_WINDOW   = 30
POST_FALL_ANGLE_THRESHOLD = 65
POST_FALL_Y_OFFSET        = 60
POST_FALL_REQUIRED_FRAMES = 6
RECOVER_ANGLE_THRESHOLD   = 35
RECOVER_TIME_SEC          = 2.0
DELTA_SUDUT_WINDOW        = 30
EMA_ALPHA                 = 0.7

# Interval pengecekan koneksi backend secara berkala (detik)
BACKEND_CHECK_INTERVAL_SEC = 5

# Palet "jadul" Windows klasik (abu-abu pucat / off-white)
WIN_FACE   = "#ECE9D8"   # abu-abu pucat khas Windows XP classic / 98
WIN_LIGHT  = "#FFFFFF"
WIN_BORDER = "#ACA899"
WIN_SHADOW = "#716F64"
WIN_BLACK  = "#000000"
WIN_WARN_BG = "#FFF3B0"
WIN_WARN_BORDER = "#C4A000"


def hitung_fitur(titik, conf_titik):
    c_bahu = min(conf_titik[5].item(), conf_titik[6].item())
    c_pinggul = min(conf_titik[11].item(), conf_titik[12].item())
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
    return {"sudut": sudut, "posisi_y": posisi_y}


class DetectionWorker(QThread):
    frame_ready    = Signal(np.ndarray)
    stats_ready    = Signal(dict)
    backend_error  = Signal(str)

    def __init__(self, camera_id=0, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._running = False

    def stop(self):
        self._running = False

    def _kirim_fall_event_ke_backend(self, frame, confidence_score=None):
        def _kirim():
            try:
                event_time = time.strftime("%Y-%m-%d %H:%M:%S")
                data = {"camera_id": str(self.camera_id), "event_time": event_time}
                if confidence_score is not None:
                    data["confidence_score"] = str(round(confidence_score, 4))
                files = {}
                if frame is not None:
                    _, buffer = cv2.imencode(".jpg", frame)
                    files["snapshot"] = ("snapshot.jpg", buffer.tobytes(), "image/jpeg")
                resp = requests.post(
                    f"{BACKEND_URL}/api/events/fall",
                    headers={"x-api-key": AI_API_KEY},
                    data=data, files=files if files else None, timeout=10
                )
                if resp.status_code != 201:
                    self.backend_error.emit(f"Fall event gagal: {resp.status_code}")
            except Exception as e:
                self.backend_error.emit(f"Error kirim fall event: {e}")
        threading.Thread(target=_kirim, daemon=True).start()

    def _kirim_status_frame_ke_backend(self, frame):
        def _kirim():
            try:
                _, buffer = cv2.imencode(".jpg", frame)
                files = {"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                resp = requests.post(
                    f"{BACKEND_URL}/api/cameras/{self.camera_id}/status-frame",
                    headers={"x-api-key": AI_API_KEY}, files=files, timeout=10
                )
                if resp.status_code == 200:
                    self.backend_error.emit(f"Frame terkirim OK (camera_id={self.camera_id})")
                else:
                    self.backend_error.emit(f"Frame gagal: {resp.status_code} - {resp.text}")
            except Exception as e:
                self.backend_error.emit(f"Error kirim frame: {e}")
        threading.Thread(target=_kirim, daemon=True).start()

    def run(self):
        self._running = True
        model_pose = YOLO("yolov8s-pose.pt")
        model_rf = joblib.load("model_fall_detection.pkl")
        kamera = cv2.VideoCapture(self.camera_id)

        if not kamera.isOpened():
            self.backend_error.emit(f"Tidak bisa membuka kamera {self.camera_id}")
            self._running = False
            return

        posisi_sebelumnya = None
        posisi_smooth = None
        waktu_sebelumnya = None
        waktu_kirim_frame_terakhir = 0

        riwayat_prediksi = []
        riwayat_kecepatan_max = []
        riwayat_kecepatan = []
        riwayat_post_fall = []
        riwayat_delta_sudut = []

        fall_state = "SAFE"
        recover_start_time = None
        baseline_posisi_y = None
        sudut_sebelumnya = None
        fall_sudah_dikirim = False

        while self._running:
            berhasil, frame = kamera.read()
            if not berhasil:
                break

            hasil = model_pose(frame, verbose=False)
            keypoints = hasil[0].keypoints
            boxes = hasil[0].boxes

            sudut = 0
            prediksi_conf = 0
            prediksi_label = "normal"
            kecepatan = 0
            kecepatan_filtered = 0
            kecepatan_puncak = 0
            delta_sudut_puncak = 0

            sekarang = time.time()
            best_idx, best_conf = None, 0.3

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

                        posisi_smooth = fitur["posisi_y"] if posisi_smooth is None else (
                            EMA_ALPHA * posisi_smooth + (1 - EMA_ALPHA) * fitur["posisi_y"]
                        )

                        kecepatan_filtered = 0
                        delta_sudut = 0
                        if posisi_sebelumnya is not None and waktu_sebelumnya is not None:
                            dt = sekarang - waktu_sebelumnya
                            if dt > 0:
                                kecepatan = (posisi_smooth - posisi_sebelumnya) / dt
                                riwayat_kecepatan.append(kecepatan)
                                if len(riwayat_kecepatan) > 5:
                                    riwayat_kecepatan.pop(0)
                                kecepatan_filtered = np.median(riwayat_kecepatan)
                                if dt <= IMPACT_WINDOW_SEC and sudut_sebelumnya is not None:
                                    delta_sudut = abs(sudut - sudut_sebelumnya)

                        riwayat_delta_sudut.append(delta_sudut)
                        if len(riwayat_delta_sudut) > DELTA_SUDUT_WINDOW:
                            riwayat_delta_sudut.pop(0)
                        delta_sudut_puncak = max(riwayat_delta_sudut) if riwayat_delta_sudut else 0

                        X_input = np.array([[sudut, kecepatan_filtered, delta_sudut_puncak]])
                        prediksi_label = model_rf.predict(X_input)[0]
                        prediksi_proba = model_rf.predict_proba(X_input)[0]
                        prediksi_conf = max(prediksi_proba)

                        riwayat_kecepatan_max.append(kecepatan_filtered)
                        if len(riwayat_kecepatan_max) > KECEPATAN_PUNCAK_WINDOW:
                            riwayat_kecepatan_max.pop(0)
                        kecepatan_puncak = max(riwayat_kecepatan_max)

                        prediksi_binary = 1 if (
                            prediksi_label == "fall" and prediksi_conf > RF_FALL_CONF_THRESHOLD
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

                        if fall_state == "SAFE" and sudut < 45:
                            baseline_posisi_y = posisi_smooth if baseline_posisi_y is None else (
                                0.9 * baseline_posisi_y + 0.1 * posisi_smooth
                            )

                        post_fall_condition = (
                            sudut > POST_FALL_ANGLE_THRESHOLD or
                            (baseline_posisi_y is not None and
                             posisi_smooth > (baseline_posisi_y + POST_FALL_Y_OFFSET))
                        )

                        if fall_state == "SAFE":
                            if impact_terdeteksi:
                                fall_state = "SUSPECT_FALL"
                                recover_start_time = None
                                riwayat_post_fall = []

                        elif fall_state == "SUSPECT_FALL":
                            riwayat_post_fall.append(1 if post_fall_condition else 0)
                            if len(riwayat_post_fall) > 30:
                                riwayat_post_fall.pop(0)
                            if sum(riwayat_post_fall) >= POST_FALL_REQUIRED_FRAMES:
                                fall_state = "FALL_CONFIRMED"

                        elif fall_state == "FALL_CONFIRMED":
                            if sudut < RECOVER_ANGLE_THRESHOLD:
                                if recover_start_time is None:
                                    recover_start_time = sekarang
                                elif (sekarang - recover_start_time) >= RECOVER_TIME_SEC:
                                    fall_state = "SAFE"
                                    recover_start_time = None
                                    riwayat_post_fall = []
                                    riwayat_prediksi = []
                                    riwayat_kecepatan_max = []
                                    riwayat_kecepatan = []
                                    riwayat_delta_sudut = []
                                    fall_sudah_dikirim = False

                        if sudut < 30:
                            riwayat_prediksi = []
                            riwayat_kecepatan_max = []
                            riwayat_kecepatan = []
                            riwayat_delta_sudut = []

                        posisi_sebelumnya = posisi_smooth
                        waktu_sebelumnya = sekarang
                        sudut_sebelumnya = sudut
            else:
                posisi_sebelumnya = None
                waktu_sebelumnya = None
                sudut_sebelumnya = None

            frame_annotated = hasil[0].plot()

            if fall_state == "FALL_CONFIRMED" and not fall_sudah_dikirim:
                self._kirim_fall_event_ke_backend(frame_annotated, confidence_score=prediksi_conf)
                fall_sudah_dikirim = True

            if (sekarang - waktu_kirim_frame_terakhir) >= JEDA_KIRIM_FRAME:
                self._kirim_status_frame_ke_backend(frame_annotated)
                waktu_kirim_frame_terakhir = sekarang

            if fall_state == "SAFE":
                status_text = "AMAN"
            elif fall_state == "SUSPECT_FALL":
                status_text = "TERDUGA JATUH"
            else:
                status_text = "JATUH TERDETEKSI!"

            self.frame_ready.emit(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))
            self.stats_ready.emit({
                "status": status_text,
                "level": fall_state,
                "sudut": sudut,
                "kecepatan": kecepatan,
                "kecepatan_filtered": kecepatan_filtered,
                "kecepatan_puncak": kecepatan_puncak,
                "rf_label": str(prediksi_label),
                "rf_conf": float(prediksi_conf),
                "impact_vote": sum(riwayat_prediksi),
                "impact_vote_max": IMPACT_VOTE_WINDOW,
                "post_fall": sum(riwayat_post_fall),
                "post_fall_max": POST_FALL_REQUIRED_FRAMES,
                "delta_sudut_puncak": delta_sudut_puncak,
            })

        kamera.release()


class StartupBackendCheck(QThread):
    checked = Signal(bool)

    def _check(self):
        try:
            resp = requests.get(BACKEND_URL, timeout=5)
            return resp.status_code < 500
        except Exception:
            return False

    def run(self):
        ok = self._check()
        self.checked.emit(ok)


class CameraView(QWidget):
    """Jendela kamera dengan label status/sudut/kecepatan mengambang di atasnya."""

    LEVEL_COLOR = {
        "SAFE": "#008000",
        "SUSPECT_FALL": "#B8860B",
        "FALL_CONFIRMED": "#FF0000",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(WIN_BLACK))
        self.setPalette(pal)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background-color: {WIN_BLACK};")
        lay.addWidget(self.image_label)

        # --- overlay label status utama (AMAN / TERDUGA JATUH / JATUH TERDETEKSI) ---
        self.status_overlay = QLabel(self)
        self.status_overlay.setFont(QFont("MS Shell Dlg 2", 10, QFont.Bold))
        self.status_overlay.setStyleSheet(
            f"background-color: {WIN_FACE}; color: {WIN_BLACK};"
            f"border: 1px solid {WIN_BORDER}; padding: 3px 8px;"
        )
        self.status_overlay.hide()

        # --- overlay semua metrics (sudut, kecepatan, RF, vote, dst) ---
        # sekarang diletakkan di pojok KIRI BAWAH, bukan kiri atas
        self.metrics_overlay = QLabel(self)
        self.metrics_overlay.setFont(QFont("Courier New", 9))
        self.metrics_overlay.setStyleSheet(
            f"background-color: {WIN_FACE}; color: {WIN_BLACK};"
            f"border: 1px solid {WIN_BORDER}; padding: 4px 8px;"
        )
        self.metrics_overlay.hide()

        self._reposition_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlays()

    def _reposition_overlays(self):
        margin = 8

        # status utama tetap di pojok kiri ATAS
        self.status_overlay.adjustSize()
        self.status_overlay.move(margin, margin)

        # container parameter AI (metrics) sekarang di pojok kiri BAWAH
        self.metrics_overlay.adjustSize()
        x = margin
        y = self.height() - self.metrics_overlay.height() - margin
        self.metrics_overlay.move(x, max(margin, y))

    def set_frame(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def set_stats(self, s: dict):
        self.status_overlay.setText(s.get("status", ""))
        self.status_overlay.setStyleSheet(
            f"background-color: {WIN_FACE}; "
            f"color: {self.LEVEL_COLOR.get(s.get('level'), WIN_BLACK)};"
            f"border: 1px solid {WIN_BORDER}; padding: 3px 8px; font-weight: bold;"
        )

        lines = [
            f"Sudut            : {s.get('sudut', 0):.1f}",
            f"Kecepatan        : {s.get('kecepatan', 0):.1f}",
            f"Kecepatan Filter : {s.get('kecepatan_filtered', 0):.1f}",
            f"Kecepatan Puncak : {s.get('kecepatan_puncak', 0):.1f}",
            f"RF               : {s.get('rf_label', '-')} ({s.get('rf_conf', 0):.1%})",
            f"Impact Vote      : {s.get('impact_vote', 0)}/{s.get('impact_vote_max', 0)}",
            f"Post-fall        : {s.get('post_fall', 0)}/{s.get('post_fall_max', 0)}",
            f"Delta Sudut Pk   : {s.get('delta_sudut_puncak', 0):.1f}",
        ]
        self.metrics_overlay.setText("\n".join(lines))

        self.status_overlay.show()
        self.metrics_overlay.show()
        self.status_overlay.raise_()
        self.metrics_overlay.raise_()
        self._reposition_overlays()

    def clear_frame(self):
        self.image_label.clear()
        self.status_overlay.hide()
        self.metrics_overlay.hide()


class SentinelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SentinelAI")
        self.resize(960, 560)
        self.worker = None
        self.backend_connected = None  # None = belum pernah dicek

        self._apply_classic_palette()
        self._build_ui()

        # --- cek koneksi backend SEKALI saja saat startup ---
        self.startup_check = StartupBackendCheck()
        self.startup_check.checked.connect(self.on_startup_backend_checked)
        self.startup_check.start()

    def _apply_classic_palette(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(WIN_FACE))
        pal.setColor(QPalette.Button, QColor(WIN_FACE))
        pal.setColor(QPalette.ButtonText, QColor(WIN_BLACK))
        pal.setColor(QPalette.WindowText, QColor(WIN_BLACK))
        self.setPalette(pal)
        self.setFont(QFont("MS Shell Dlg 2", 9))

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root.setAutoFillBackground(True)
        pal = root.palette()
        pal.setColor(QPalette.Window, QColor(WIN_FACE))
        root.setPalette(pal)

        root_lay = QHBoxLayout(root)
        root_lay.setContentsMargins(10, 10, 10, 10)
        root_lay.setSpacing(10)

        # 1. MAIN CAMERA WINDOW
        self.camera_view = CameraView()
        frame = QFrame()
        frame.setFrameShape(QFrame.Panel)
        frame.setFrameShadow(QFrame.Sunken)
        frame.setLineWidth(2)
        frame_lay = QVBoxLayout(frame)
        frame_lay.setContentsMargins(2, 2, 2, 2)
        frame_lay.addWidget(self.camera_view)
        root_lay.addWidget(frame, 3)

        # Panel kanan
        right_panel = QFrame()
        right_panel.setFixedWidth(180)
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(8)
        right_lay.addStretch()

        # 2. TOMBOL START
        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedHeight(32)
        self.start_btn.clicked.connect(self.start_monitoring)
        right_lay.addWidget(self.start_btn)

        # 3. TOMBOL STOP
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        right_lay.addWidget(self.stop_btn)

        # 4. TOMBOL CONNECT BACKEND
        self.connect_btn = QPushButton("Connect Backend")
        self.connect_btn.setFixedHeight(32)
        self.connect_btn.clicked.connect(self.connect_backend)
        right_lay.addWidget(self.connect_btn)

        right_lay.addStretch()
        root_lay.addWidget(right_panel, 0)

    # ---------------------------------------------------
    def start_monitoring(self):
        self.worker = DetectionWorker(camera_id=0)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats_ready.connect(self.camera_view.set_stats)
        # error jaringan per-request sudah tidak memicu popup lagi;
        # status koneksi backend cukup ditangani oleh BackendChecker + banner
        self.worker.backend_error.connect(self.on_backend_error_silent)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_monitoring(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_view.clear_frame()

    def connect_backend(self):
        try:
            resp = requests.get(BACKEND_URL, timeout=5)
            ok = resp.status_code < 500
        except Exception:
            ok = False

        self.backend_connected = ok

        box = QMessageBox(self)
        box.setWindowTitle("SentinelAI")
        if ok:
            box.setIcon(QMessageBox.Information)
            box.setText("Backend telah terkoneksi.")
        else:
            box.setIcon(QMessageBox.Critical)
            box.setText("Backend tidak dapat terkoneksi.")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    # ---------------------------------------------------
    @Slot(np.ndarray)
    def on_frame(self, frame_rgb):
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_view.set_frame(qimg.copy())

    @Slot(bool)
    def on_startup_backend_checked(self, ok: bool):
        # Dijalankan cuma SEKALI di awal (saat app dibuka).
        # Kalau backend nyala -> tidak menampilkan apa-apa.
        # Kalau backend mati -> munculkan satu popup peringatan, selesai.
        self.backend_connected = ok
        if not ok:
            box = QMessageBox(self)
            box.setWindowTitle("SentinelAI")
            box.setIcon(QMessageBox.Warning)
            box.setText("Backend belum nyala.")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec()

    @Slot(str)
    def on_backend_error_silent(self, text):
        # Error individual dari worker (mis. gagal kirim satu frame) tidak lagi
        # memunculkan popup. Koneksi backend sudah dipantau terus-menerus oleh
        # BackendChecker dan ditampilkan lewat banner non-modal di atas.
        # Kalau butuh debugging, cetak saja ke konsol:
        print(f"[backend] {text}")

    def closeEvent(self, event):
        self.stop_monitoring()
        if hasattr(self, "startup_check") and self.startup_check.isRunning():
            self.startup_check.wait(1000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SentinelWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()