import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import tempfile
import threading
import time

# Load model YOLOv8 dengan error handling
try:
    model = YOLO("best.pt")  # Pastikan ini model ayam & kalkun-mu
    print("Model berhasil dimuat!")
    print(f"Classes yang tersedia: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Pastikan file 'best.pt' ada di folder yang sama dengan script ini")
    exit(1)

# Global variables untuk webcam streaming
webcam_active = False
current_frame = None

def detect_image(image):
    """
    Fungsi untuk deteksi objek pada gambar yang diupload
    """
    try:
        if image is None:
            return None, "Tidak ada gambar yang diupload"
        
        # Konversi PIL Image ke array numpy
        img_array = np.array(image)
        
        # Lakukan inference
        results = model(img_array)[0]
        
        # Plot hasil deteksi
        annotated = results.plot()
        
        # Konversi BGR ke RGB untuk ditampilkan di Gradio
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Hitung jumlah deteksi dengan penanganan error yang lebih baik
        detections = 0
        summary = "Tidak ada objek yang terdeteksi"
        
        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            detections = len(results.boxes)
            summary = f"Terdeteksi {detections} objek"
            
            try:
                # Ambil class labels dengan penanganan error
                classes = results.boxes.cls.cpu().numpy()
                class_names = []
                
                for cls in classes:
                    cls_int = int(cls)
                    if cls_int in model.names:
                        class_names.append(model.names[cls_int])
                    else:
                        class_names.append(f"Unknown_{cls_int}")
                
                # Hitung jumlah setiap class
                class_counts = {}
                for name in class_names:
                    if name in class_counts:
                        class_counts[name] += 1
                    else:
                        class_counts[name] = 1
                
                if class_counts:
                    summary += "\nDetail:\n"
                    for class_name, count in class_counts.items():
                        summary += f"- {class_name}: {count}\n"
                        
            except Exception as e:
                summary += f"\n(Error dalam detail klasifikasi: {str(e)})"
        
        return annotated_rgb, summary
        
    except Exception as e:
        error_msg = f"Error dalam deteksi gambar: {str(e)}"
        print(error_msg)  # Untuk debugging
        return None, error_msg

def detect_video(video_path):
    """
    Fungsi untuk deteksi objek pada video
    """
    try:
        if video_path is None:
            return None, "Tidak ada video yang diupload"
        
        # Baca video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Gagal membuka video"
        
        # Dapatkan properti video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            return None, "Video properties tidak valid"
        
        # Buat video output sementara
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Lakukan inference
                results = model(frame)[0]
                annotated = results.plot()
                
                # Tulis frame ke video output
                out.write(annotated)
                
                # Hitung deteksi
                if hasattr(results, 'boxes') and results.boxes is not None:
                    total_detections += len(results.boxes)
                
                frame_count += 1
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Tulis frame asli jika ada error
                out.write(frame)
                frame_count += 1
        
        cap.release()
        out.release()
        
        summary = f"Video diproses: {frame_count} frame\nTotal deteksi: {total_detections} objek"
        
        return temp_output.name, summary
        
    except Exception as e:
        error_msg = f"Error dalam memproses video: {str(e)}"
        print(error_msg)
        return None, error_msg

def webcam_inference(image):
    """
    Fungsi untuk inference real-time langsung pada webcam stream
    Dipanggil otomatis setiap frame webcam berubah
    """
    try:
        if image is None:
            return None
        
        # Konversi ke numpy array jika belum
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Lakukan inference
        results = model(img_array)[0]
        
        # Plot hasil deteksi langsung pada frame
        annotated = results.plot()
        
        # Konversi BGR ke RGB untuk ditampilkan di Gradio
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb
        
    except Exception as e:
        print(f"Error dalam inference real-time: {e}")
        # Return frame asli jika ada error
        if isinstance(image, Image.Image):
            return np.array(image)
        return image

def get_detection_summary(image):
    """
    Fungsi untuk mendapatkan ringkasan deteksi
    """
    try:
        if image is None:
            return "Tidak ada frame dari webcam"
        
        # Konversi ke numpy array jika belum
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Lakukan inference
        results = model(img_array)[0]
        
        # Hitung jumlah deteksi
        detections = 0
        summary = "🔴 LIVE - Tidak ada objek terdeteksi"
        
        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            detections = len(results.boxes)
            summary = f"🔴 LIVE - Terdeteksi {detections} objek"
            
            try:
                # Ambil class labels
                classes = results.boxes.cls.cpu().numpy()
                class_names = []
                
                for cls in classes:
                    cls_int = int(cls)
                    if cls_int in model.names:
                        class_names.append(model.names[cls_int])
                    else:
                        class_names.append(f"Unknown_{cls_int}")
                
                # Hitung jumlah setiap class
                class_counts = {}
                for name in class_names:
                    if name in class_counts:
                        class_counts[name] += 1
                    else:
                        class_counts[name] = 1
                
                if class_counts:
                    summary += "\n📊 Detail Real-time:\n"
                    for class_name, count in class_counts.items():
                        summary += f"🐔 {class_name}: {count}\n"
                        
            except Exception as e:
                summary += f"\n(Error dalam detail: {str(e)})"
        
        # Tambahkan timestamp
        current_time = time.strftime("%H:%M:%S")
        summary += f"\n⏰ Update: {current_time}"
        
        return summary
        
    except Exception as e:
        return f"Error dalam analisis: {str(e)}"

# CSS untuk styling
css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    color: #2E8B57;
    margin-bottom: 30px;
}
.live-indicator {
    background-color: #ff4444;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 10px;
}
"""

# Buat interface dengan Gradio
with gr.Blocks(css=css, title="🐔 Deteksi Ayam & Kalkun") as app:
    gr.Markdown("# 🐔 Sistem Deteksi Ayam & Kalkun Real-time 🦃")
    
    with gr.Tabs():
        # Tab untuk upload gambar
        with gr.TabItem("📷 Upload Gambar"):
            gr.Markdown("### Upload gambar untuk mendeteksi ayam dan kalkun")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Gambar")
                    detect_btn = gr.Button("🔍 Deteksi Objek", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Hasil Deteksi")
                    summary_output = gr.Textbox(label="Ringkasan Deteksi", lines=5)
            
            detect_btn.click(
                fn=detect_image,
                inputs=image_input,
                outputs=[image_output, summary_output]
            )
        
        # Tab untuk upload video
        with gr.TabItem("🎥 Upload Video"):
            gr.Markdown("### Upload video untuk mendeteksi ayam dan kalkun")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    process_video_btn = gr.Button("🎬 Proses Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Video Hasil Deteksi")
                    video_summary = gr.Textbox(label="Ringkasan Video", lines=3)
            
            process_video_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=[video_output, video_summary]
            )
        
        # Tab untuk real-time webcam detection (True Real-time)
        with gr.TabItem("📹 Real-time Webcam"):
            gr.Markdown("""
            ### 🔴 TRUE LIVE Real-time Detection
            **Deteksi berjalan otomatis tanpa tombol!** Webcam langsung menampilkan bounding box secara real-time.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📱 Live Camera with Real-time Detection")
                    # Webcam dengan inference langsung
                    webcam_live = gr.Interface(
                        fn=webcam_inference,
                        inputs=gr.Image(source="webcam", streaming=True, label="🔴 LIVE Detection Camera"),
                        outputs=gr.Image(label="🎯 Real-time Output with Bounding Boxes"),
                        live=True,  # Ini yang membuat real-time tanpa tombol
                        allow_flagging="never",
                        title="",
                        description=""
                    )
                
                with gr.Column():
                    gr.Markdown("#### 📊 Live Detection Info")
                    # Interface terpisah untuk summary
                    summary_interface = gr.Interface(
                        fn=get_detection_summary,
                        inputs=gr.Image(source="webcam", streaming=True, label="🔄 Summary Camera"),
                        outputs=gr.Textbox(label="📈 Live Detection Summary", lines=8),
                        live=True,
                        allow_flagging="never",
                        title="",
                        description=""
                    )
        
        # Tab untuk webcam manual (backup option)
        with gr.TabItem("📸 Manual Webcam"):
            gr.Markdown("### Manual webcam detection — klik tombol untuk deteksi")

            with gr.Row():
                with gr.Column():
                    webcam_input_manual = gr.Image(source="webcam", streaming=True, label="Manual Webcam Stream")
                    detect_webcam_btn = gr.Button("📸 Ambil & Deteksi", variant="primary")
                
                with gr.Column():
                    webcam_result_img = gr.Image(label="Hasil Deteksi")
                    webcam_result_text = gr.Textbox(label="Ringkasan Deteksi", lines=5)

            def detect_from_webcam_frame(image):
                """Deteksi saat tombol ditekan dari stream webcam"""
                if image is not None:
                    return detect_image(Image.fromarray(image))
                return None, "Tidak ada frame yang tersedia"

            detect_webcam_btn.click(
                fn=detect_from_webcam_frame,
                inputs=webcam_input_manual,
                outputs=[webcam_result_img, webcam_result_text]
            )
    
    # Footer
    gr.Markdown(
        """
        ---
        ### 🔧 Informasi Teknis
        - Model: YOLOv8 Custom (best.pt)
        - Objek yang dapat dideteksi: Ayam dan Kalkun
        - Format gambar yang didukung: JPG, PNG, WEBP
        - Format video yang didukung: MP4, AVI, MOV
        
        ### 📹 Cara Menggunakan:
        **🔴 Real-time Webcam (TRUE LIVE):**
        - Webcam langsung menampilkan bounding box secara real-time
        - Tidak perlu menekan tombol apapun
        - Deteksi berjalan otomatis saat ada objek di kamera
        - Bounding box muncul langsung pada video stream
        
        **📸 Manual Webcam (Backup):**
        - Klik tombol "Ambil & Deteksi" untuk deteksi manual
        - Berguna jika real-time terlalu berat untuk perangkat
        
        ### ⚡ Tips Performa:
        - Pastikan pencahayaan yang cukup
        - Jaga kamera tetap stabil
        - Real-time detection membutuhkan koneksi dan perangkat yang baik
        - Gunakan browser Chrome/Firefox terbaru untuk performa optimal
        """
    )

if __name__ == '__main__':
    # Jalankan aplikasi
    app.launch(
        share=True,             # Set True untuk public link
        debug=True,
        inbrowser=False         # Tidak buka browser otomatis
    )