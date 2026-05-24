import streamlit as st
import signal

# Monkeypatch signal.signal to work in Streamlit's multi-threaded environment
def nosection(*args, **kwargs):
    pass
signal.signal = nosection

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import tempfile
import os
import time
import re
from collections import Counter

# --- Page Config ---
st.set_page_config(
    page_title="ANPR Pro - YOLOv8",
    page_icon="🚗",
    layout="wide"
)

# --- Load Styles ---
with open("app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Constants ---
MODEL_PATH = "ultralytics/yolo/v8/detect/best.pt"

# --- Initialization ---
@st.cache_resource
def load_models():
    # Load YOLOv8
    model = YOLO(MODEL_PATH)
    # Load EasyOCR
    reader = easyocr.Reader(['en'], gpu=torch_cuda if (torch_cuda := cv2.cuda.getCudaEnabledDeviceCount() > 0) else False)
    return model, reader

def get_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def clean_text(text):
    """Remove non-alphanumeric characters and clean up the plate text, including stripping 'IND'."""
    text = text.upper().strip()
    # Remove "IND" if it appears at the start (common in Indian plates)
    if text.startswith("IND"):
        text = text[3:].strip()
    # Also handle cases where IND might be stuck to the next char
    text = re.sub(r'^IND', '', text)
    return re.sub(r'[^A-Z0-9]', '', text)

def ocr_plate(img, box, reader):
    x1, y1, x2, y2 = map(int, box)
    # Ensure coordinates are within image bounds and add small padding
    h, w = img.shape[:2]
    padding = 4
    x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(w, x2+padding), min(h, y2+padding)
    
    # Crop plate
    plate_img = img[y1:y2, x1:x2]
    if plate_img.size == 0:
        return ""
    
    # --- IND Removal: Crop Left 10% ---
    # Many Indian plates have "IND" and a hologram on the far left.
    # Cropping it helps avoid OCR noise.
    h_p, w_p = plate_img.shape[:2]
    plate_img = plate_img[:, int(w_p * 0.10):]
    
    # --- Enhanced Preprocessing ---
    # 1. Upscale for better OCR (especially for small bike plates)
    plate_img = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Denoising
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # EasyOCR
    results = reader.readtext(gray)
    
    # Sort results by vertical position (y-coordinate) to handle 2-line plates (Motorcycles)
    # res[0] is the bounding box [ [x1,y1], [x2,y1], [x2,y2], [x1,y2] ]
    results.sort(key=lambda x: x[0][0][1]) 
    
    text = ""
    for res in results:
        if res[2] > 0.15: # Slightly lower threshold to catch bike plate chars
            text += res[1].upper() + " "
    
    cleaned = clean_text(text.strip())
    # Filter out very short strings
    return cleaned if len(cleaned) > 3 else ""

# --- UI Components ---
st.markdown('<h1 class="main-title">Automatic Number Plate Recognition</h1>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
frame_skip = st.sidebar.slider("Frame Skip (Video Only)", 1, 10, 3)
st.sidebar.markdown("---")
st.sidebar.info("This application uses YOLOv8 for plate detection and EasyOCR for character recognition.")

# --- Execution ---
try:
    model, reader = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

tab1, tab2 = st.tabs(["🖼️ Image Processing", "🎥 Video Processing"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("🚀 Process Image"):
            with st.spinner("Detecting and Recognizing..."):
                start_time = time.time()
                results = model.predict(img_bgr, conf=conf_threshold)
                
                detected_plates = []
                for result in results:
                    # In this version of YOLOv8, results are returned as raw tensors
                    for det in result:
                        coords = det[:4].cpu().numpy()
                        conf = float(det[4])
                        cls = int(det[5])
                        
                        text = ocr_plate(img_bgr, coords, reader)
                        detected_plates.append({"box": coords, "text": text})
                        
                        # Draw on image
                        cv2.rectangle(img_bgr, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
                        cv2.putText(img_bgr, f"{text}", (int(coords[0]), int(coords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                end_time = time.time()
                
            with col2:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            
            st.markdown("### 📋 Results")
            if detected_plates:
                for idx, plate in enumerate(detected_plates):
                    st.success(f"Plate {idx+1}: **{plate['text']}**")
            else:
                st.warning("No number plates detected.")
            
            st.info(f"Processing time: {end_time - start_time:.2f} seconds")

with tab2:
    video_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.video(video_file)
        
        if st.button("🎬 Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # track_history: {track_id: [list of detected texts]}
            # active_tracks: {track_id: last_box}
            track_history = {}
            active_tracks = {}
            next_id = 0
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame skipping for efficiency
                if current_frame % frame_skip != 0:
                    current_frame += 1
                    continue
                
                # Inference using predict
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                
                # In this version, results are likely a list of tensors or results objects
                detections = []
                for res in results:
                    # Based on original code, it might be a tensor of [x1, y1, x2, y2, conf, cls]
                    # or it might be a Results object. Let's handle both or stick to what worked before.
                    if hasattr(res, 'boxes'): # Newer YOLOv8 structure
                         boxes = res.boxes.xyxy.cpu().numpy()
                    else: # Raw tensor structure as seen in previous logic
                         boxes = res.cpu().numpy()
                    
                    for box in boxes:
                        detections.append(box[:4]) # Keep only coords
                
                new_active_tracks = {}
                for det_box in detections:
                    best_id = -1
                    best_iou = 0.3 # Threshold for tracking
                    
                    for tid, last_box in active_tracks.items():
                        iou = get_iou(det_box, last_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = tid
                    
                    if best_id != -1:
                        # Match found
                        track_id = best_id
                    else:
                        # New track
                        track_id = next_id
                        next_id += 1
                        track_history[track_id] = []
                    
                    new_active_tracks[track_id] = det_box
                    
                    # Process OCR
                    text = ocr_plate(frame, det_box, reader)
                    if text:
                        track_history[track_id].append(text)
                    
                    # Draw
                    cv2.rectangle(frame, (int(det_box[0]), int(det_box[1])), (int(det_box[2]), int(det_box[3])), (0, 255, 0), 2)
                    label = f"ID: {track_id} {text}"
                    cv2.putText(frame, label, (int(det_box[0]), int(det_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                active_tracks = new_active_tracks

                # Show frame
                st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                current_frame += 1
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                
            cap.release()
            st.success("Video processing complete!")
            
            # Aggregate results: for each track ID, pick the most common OCR result
            final_results = []
            for track_id, texts in track_history.items():
                if texts:
                    # Filter out short or junk results if necessary, then pick most common
                    most_common_text = Counter(texts).most_common(1)[0][0]
                    final_results.append(most_common_text)
            
            if final_results:
                st.markdown("### 📋 All Detected Number Plates")
                cols = st.columns(3)
                for i, num in enumerate(sorted(list(set(final_results)))):
                    cols[i % 3].info(f"**{num}**")
            else:
                st.warning("No number plates were detected in the video.")

st.markdown("---")
st.markdown("Developed with ❤️ using YOLOv8 & Streamlit")
