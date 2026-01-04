import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import tempfile
import os
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="👷", layout="centered")
st.title("👷 Sistem Deteksi K3: Helm Keselamatan")
st.write("Aplikasi ini menggunakan YOLOv8 untuk mendeteksi penggunaan helm pada pekerja.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('helm_yolo.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# --- PILIHAN MENU ---
st.write("---")
option = st.selectbox("Pilih Metode Deteksi:", 
                     ("Live Kamera (HP/Laptop)", "Ambil Foto (Alternatif)", "Upload Video"))

# ==========================================
# 1. LIVE KAMERA (ANTI GEPENG)
# ==========================================
if option == "Live Kamera (HP/Laptop)":
    st.info("💡 Support mode Portrait (HP) & Landscape (Laptop).")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # --- PERBAIKAN: SMART RESIZE ---
        h_asli, w_asli = img.shape[:2]
        target_width = 600
        ratio = target_width / float(w_asli)
        target_height = int(h_asli * ratio)
        
        img_resized = cv2.resize(img, (target_width, target_height))
        
        results = model(img_resized, conf=0.6)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    webrtc_streamer(
        key="safety-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==========================================
# 2. AMBIL FOTO (BACKUP)
# ==========================================
elif option == "Ambil Foto (Alternatif)":
    st.write("### 📸 Mode Foto Statis")
    picture = st.camera_input("Ambil Foto")
    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        results = model.predict(cv2_img, conf=0.5)
        frame_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_column_width=True)

# ==========================================
# 3. UPLOAD VIDEO (ANTI GEPENG + FIX SYNTAX)
# ==========================================
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload video (MP4)...", type=['mp4'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        stop_btn = st.button("⏹️ Stop")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # --- BAGIAN YANG TADI ERROR (SUDAH DIPERBAIKI)
