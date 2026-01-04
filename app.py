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
                     ("Live Kamera (Stream)", "Ambil Foto (Alternatif)", "Upload Video"))

# ==========================================
# 1. LIVE KAMERA (WebRTC) - DENGAN STUN LEBIH BANYAK
# ==========================================
if option == "Live Kamera (Stream)":
    st.info("💡 Jika loading terus, berarti jaringan memblokir. Silakan pindah ke menu 'Ambil Foto'.")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (640, 480))
        
        results = model(img_resized, conf=0.5)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Konfigurasi Server STUN Ganda (Biar Tembus Firewall)
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
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
# 2. AMBIL FOTO (SOLUSI ANTI-MACET)
# ==========================================
elif option == "Ambil Foto (Alternatif)":
    st.write("### 📸 Mode Foto Statis")
    st.warning("Gunakan mode ini jika Live Stream tidak muncul karena gangguan sinyal/firewall.")
    
    picture = st.camera_input("Ambil Foto untuk Deteksi")
    
    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        results = model.predict(cv2_img, conf=0.5)
        res_plotted = results[0].plot()
        
        frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Hasil Deteksi", use_column_width=True)

# ==========================================
# 3. UPLOAD VIDEO (YANG SUDAH DIPERBAIKI)
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
            if not ret or stop_btn:
                break
            
            frame = cv2.resize(frame, (640, 480))
            results = model.predict(frame, conf=0.45, verbose=False)
            res_plotted = results[0].plot()
            
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()
        try: os.unlink(tfile.name) 
        except: pass
