import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="👷")
st.title("👷 Sistem Deteksi K3: Helm Keselamatan")
st.write("Aplikasi ini menggunakan YOLOv8 untuk mendeteksi penggunaan helm pada pekerja.")

# --- LOAD MODEL ---
MODEL_PATH = 'helm_yolo.pt' 

if not os.path.exists(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Pastikan file ada di folder yang sama!")
else:
    model = YOLO(MODEL_PATH)

    # --- PILIHAN SUMBER ---
    option = st.selectbox("Pilih Sumber Deteksi:", ("Upload Video", "Gunakan Webcam"))

    # ==========================================
    # LOGIKA UPLOAD VIDEO
    # ==========================================
    if option == "Upload Video":
        uploaded_file = st.file_uploader("Upload video proyek (mp4/avi)...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Tombol Stop untuk Video
            stop_button = st.button("⏹️ Stop Video")
            
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break
                
                # Deteksi
                results = model.predict(frame, conf=0.45)
                res_plotted = results[0].plot()
                
                # Tampilkan
                frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
            
            cap.release()

    # ==========================================
    # LOGIKA WEBCAM (FITUR STOP DITAMBAHKAN)
    # ==========================================
    elif option == "Gunakan Webcam":
        st.write("---")
        
        # Pilihan Kamera
        cam_options = {
            "Kamera Utama (Default)": 0,
            "Kamera Eksternal 1": 1,
            "Kamera Eksternal 2": 2
        }
        selected_cam_name = st.selectbox("Pilih Kamera:", list(cam_options.keys()))
        cam_index = cam_options[selected_cam_name]
        
        # Checkbox Utama
        run = st.checkbox('Buka Kamera')
        
        if run:
            # === FITUR BARU: TOMBOL STOP DI SINI ===
            stop_webcam = st.button("⏹️ Stop Kamera")
            
            st_frame = st.empty()
            cap = cv2.VideoCapture(cam_index)
            
            if not cap.isOpened():
                st.error(f"Gagal membuka kamera index {cam_index}.")
            else:
                while run:
                    ret, frame = cap.read()
                    
                    # Logika Berhenti: Jika gagal baca frame ATAU tombol stop ditekan
                    if not ret or stop_webcam:
                        break
                    
                    # Deteksi
                    results = model.predict(frame, conf=0.5)
                    res_plotted = results[0].plot()
                    
                    frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB")
                
                cap.release()
                
                # Pesan konfirmasi jika tombol stop ditekan
                if stop_webcam:
                    st.success("Kamera dihentikan.")