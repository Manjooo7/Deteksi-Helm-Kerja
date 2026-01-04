import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import tempfile
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="👷", layout="centered")
st.title("👷 Real-Time Safety Detection")

# --- CSS BIAR RAPI ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (CACHE BIAR CEPAT) ---
@st.cache_resource
def load_model():
    return YOLO('helm_yolo.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# --- PILIHAN MENU ---
option = st.selectbox("Pilih Metode Deteksi:", ("Live Kamera (WebRTC)", "Upload Video"))

# ==========================================
# 1. LIVE KAMERA (WebRTC - Support Cloud)
# ==========================================
if option == "Live Kamera (WebRTC)":
    st.write("### 📹 Live Streaming Deteksi")
    st.info("Gunakan fitur ini untuk deteksi langsung dari Webcam/HP.")

    # Callback: Fungsi yang dijalankan berulang-ulang untuk setiap frame kamera
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Deteksi dengan Confidence 0.5
        results = model(img, conf=0.5)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Komponen WebRTC
    webrtc_streamer(
        key="safety-cam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# ==========================================
# 2. UPLOAD VIDEO (Optimized / Turbo Mode)
# ==========================================
elif option == "Upload Video":
    st.write("### 📂 Analisa Rekaman Video")
    uploaded_file = st.file_uploader("Upload video (MP4/AVI)...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder gambar
        stop_btn = st.button("⏹️ Stop Video")
        
        # --- KONFIGURASI TURBO ---
        skip_frames = 2  # Lewati 2 frame, proses frame ke-3 (Meningkatkan speed 3x lipat)
        frame_counter = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
            
            frame_counter += 1
            
            # TEKNIK 1: FRAME SKIPPING
            # Jika bukan giliran frame ini, lewati saja
            if frame_counter % (skip_frames + 1) != 0:
                continue

            # TEKNIK 2: RESIZE (PENTING BANGET BUAT CLOUD)
            # Perkecil gambar jadi lebar 640px (biar ringan dihitung CPU)
            height, width = frame.shape[:2]
            new_width = 640
            new_height = int(height * (new_width / width))
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Deteksi
            results = model.predict(frame_resized, conf=0.45)
            res_plotted = results[0].plot()
            
            # Tampilkan
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
        
        cap.release()
