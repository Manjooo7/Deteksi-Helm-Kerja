import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import tempfile
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="👷", layout="centered")

# --- JUDUL ---
st.title("👷 Sistem Deteksi K3: Helm Keselamatan")
st.write("Aplikasi ini menggunakan YOLOv8 untuk mendeteksi penggunaan helm pada pekerja.")
st.markdown("---")

# --- LOAD MODEL (Cache) ---
@st.cache_resource
def load_model():
    return YOLO('helm_yolo.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# --- PILIHAN MENU ---
option = st.selectbox("Pilih Metode Deteksi:", ("Upload Video", "Live Kamera"))

# ==========================================
# 1. LIVE KAMERA (WebRTC) - Tetap sama karena sudah OK
# ==========================================
if option == "Live Kamera":
    st.info("💡 Izinkan akses kamera browser jika diminta.")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Resize sedikit biar ngebut di live
        img_resized = cv2.resize(img, (640, 480))
        
        results = model(img_resized, conf=0.5)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="safety-cam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# ==========================================
# 2. UPLOAD VIDEO (FIXED VERSION)
# ==========================================
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload video (MP4/AVI)...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Simpan file sementara dengan cara lebih aman
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        tfile.flush() # Pastikan file tertulis sempurna
        tfile.close() # Tutup dulu biar aman
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Tombol Stop
        stop_btn = st.button("⏹️ Stop Video")
        
        # Placeholder Gambar & Status
        st_frame = st.empty()
        st_status = st.empty()
        
        frame_counter = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
            
            frame_counter += 1
            
            # --- SOLUSI VIDEONYA DISINI ---
            # Kita paksa RESIZE ke lebar 480px (Standard Definition).
            # Ini akan membuat video 5x lebih ringan diproses Cloud.
            height, width = frame.shape[:2]
            target_width = 480
            target_height = int(height * (target_width / width))
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            # Deteksi
            results = model.predict(frame_resized, conf=0.45, verbose=False)
            res_plotted = results[0].plot()
            
            # Tampilkan Indikator berjalan
            if frame_counter % 10 == 0:
                st_status.text(f"Memproses frame ke-{frame_counter}...")

            # Tampilkan Gambar
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
        
        cap.release()
        st_status.text("Selesai.")
        # Hapus file sampah
        os.unlink(tfile.name)
