import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import numpy as np # Kita butuh ini untuk proses foto

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="👷")
st.title("👷 Sistem Deteksi K3: Helm Keselamatan")

# --- LOAD MODEL ---
MODEL_PATH = 'helm_yolo.pt' 

if not os.path.exists(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan.")
else:
    model = YOLO(MODEL_PATH)

    # --- PILIHAN SUMBER ---
    option = st.selectbox("Pilih Metode:", ("Upload Video", "Ambil Foto (Webcam)"))

    # ==========================================
    # 1. LOGIKA UPLOAD VIDEO (Tetap Sama)
    # ==========================================
    if option == "Upload Video":
        st.info("💡 Metode ini paling direkomendasikan untuk demo real-time.")
        uploaded_file = st.file_uploader("Upload video proyek (mp4/avi)...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stop_button = st.button("⏹️ Stop Video")
            st_frame = st.empty()
            
            # Counter untuk skip frame biar cloud gak berat
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break
                
                frame_count += 1
                if frame_count % 3 != 0: # Skip frame biar agak ngebut di cloud
                    continue

                # Deteksi
                results = model.predict(frame, conf=0.45)
                res_plotted = results[0].plot()
                
                # Tampilkan
                frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
            
            cap.release()

    # ==========================================
    # 2. LOGIKA KAMERA KHUSUS CLOUD
    # ==========================================
    elif option == "Ambil Foto (Webcam)":
        st.write("### 📸 Ambil Foto Diri")
        st.warning("Karena aplikasi berjalan di Cloud Server, fitur Live Video diganti dengan 'Ambil Foto' agar kompatibel dengan browser HP/Laptop.")
        
        # Widget Kamera bawaan Streamlit (Support Cloud)
        picture = st.camera_input("Klik tombol di bawah untuk ambil foto")
        
        if picture:
            # 1. Baca data gambar
            bytes_data = picture.getvalue()
            
            # 2. Ubah ke format yang dimengerti OpenCV
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # 3. Deteksi dengan YOLO
            results = model.predict(cv2_img, conf=0.5)
            res_plotted = results[0].plot()
            
            # 4. Tampilkan Hasil
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Hasil Deteksi AI", use_column_width=True)
