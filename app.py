import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
import sys
import time
import json
import pandas as pd
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import itertools

# --- CONFIG ---
BASE_DIR = r"C:\Users\HP\Desktop\desk"
FER_DATA_DIR = r"C:\Users\HP\Desktop\FER2013"
MODEL_PATH = os.path.join(BASE_DIR, "CNN_Model_emotion_trained.h5")
HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
if not os.path.exists(HAAR_PATH):
    HAAR_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

# --- PAGE CONFIG ---
st.set_page_config(page_title="Emotion Driving System", layout="wide")

# --- SESSION STATE INIT ---
for key in ['model', 'face_cascade', 'system_ready']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'system_ready' else False

# --- LOAD MODEL/CASCADE ---
def load_core_system():
    model_loaded, cascade_loaded = False, False
    if st.session_state.model is None and os.path.exists(MODEL_PATH):
        try:
            st.session_state.model = load_model(MODEL_PATH)
            model_loaded = True
        except Exception as e:
            st.warning(f"Model loading error: {e}")
    if st.session_state.face_cascade is None and os.path.exists(HAAR_PATH):
        fc = cv2.CascadeClassifier(HAAR_PATH)
        if not fc.empty():
            st.session_state.face_cascade = fc
            cascade_loaded = True
    st.session_state.system_ready = (st.session_state.model is not None) and (st.session_state.face_cascade is not None)
    return st.session_state.system_ready

# --- EMOTION DETECTION ---
def detect_emotions_in_frame(frame_bgr, model, face_cascade):
    results = []
    frame = frame_bgr.copy()
    if model is None or face_cascade is None or face_cascade.empty():
        cv2.putText(frame, "Model or Cascade Missing", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame, []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        try:
            roi = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        except cv2.error:
            continue
        roi_norm = roi.astype("float")/255.0
        roi_input = img_to_array(roi_norm)
        roi_input = np.expand_dims(roi_input, axis=0)
        try:
            pred = model.predict(roi_input, verbose=0)
        except Exception:
            continue
        if pred is not None and len(pred) > 0:
            idx = np.argmax(pred[0])
            label = EMOTION_LABELS[idx] if 0 <= idx < len(EMOTION_LABELS) else "Error"
            conf = pred[0][idx]
            results.append({'loc': (x, y, w, h), 'emotion': label, 'conf': float(conf), 'probs': pred[0].tolist()})
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {conf*100:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame, results

# --- STREAMLIT-WEBRTC PROCESSOR ---
class VideoProcessor:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.ready = False
        self._load()
    def _load(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
            if os.path.exists(HAAR_PATH):
                fc = cv2.CascadeClassifier(HAAR_PATH)
                if not fc.empty():
                    self.face_cascade = fc
            self.ready = (self.model is not None) and (self.face_cascade is not None)
        except Exception:
            self.ready = False
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not self.ready:
            self._load()
            if not self.ready:
                cv2.putText(img, "Processor Not Ready", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        processed, _ = detect_emotions_in_frame(img, self.model, self.face_cascade)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- PAGES ---
def page_realtime():
    st.header("📹 Real-time Webcam Emotion Detection")
    if not st.session_state.system_ready:
        st.warning("Model or face detector not loaded. WebRTC will try to load its own.")
        if st.button("Load System Components"):
            load_core_system()
            st.rerun()
    ctx = webrtc_streamer(
        key="emotion-detection-webrtc", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CFG,
        video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.state.playing:
        st.success("Webcam streaming. Detected emotions will show live.")
    else:
        st.info("Click START above to enable webcam.")

def page_img_upload():
    st.header("📸 Upload Image for Emotion Analysis")
    if not st.session_state.system_ready:
        st.warning("Model or face detector not loaded.")
        if st.button("Load System Components"):
            load_core_system()
            st.rerun()
        return
    file = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])
    if file:
        pil = Image.open(file).convert('RGB')
        np_img = np.array(pil)
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        st.image(pil, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Emotions"):
            with st.spinner("Processing..."):
                processed, results = detect_emotions_in_frame(cv_img, st.session_state.model, st.session_state.face_cascade)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Detected Emotions", use_column_width=True)
            if results:
                st.success(f"{len(results)} face(s) detected.")
                for i, res in enumerate(results):
                    with st.expander(f"Face {i+1}", expanded=True):
                        st.metric("Emotion", res['emotion'].title(), delta=f"{res['conf']:.1%}", delta_color="off")
                        prob_df = pd.DataFrame({'Emotion': EMOTION_LABELS, 'Probability': res['probs']})
                        prob_df = prob_df.sort_values('Probability', ascending=True)
                        fig = px.bar(prob_df, x='Probability', y='Emotion', orientation='h', title="Probabilities", text_auto=".2p")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No faces detected.")

def page_video_upload():
    st.header("🎞️ Upload Video for Emotion Analysis")
    if not st.session_state.system_ready:
        st.warning("Model or face detector not loaded.")
        if st.button("Load System Components"):
            load_core_system()
            st.rerun()
        return
    file = st.file_uploader("Choose a video file", type=['mp4','avi','mov','mkv'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
        tfile.write(file.read()); video_file_path = tfile.name; tfile.close()
        st.video(video_file_path)
        if st.button("Analyze Video"):
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                st.error("Error opening video.")
                os.unlink(video_file_path)
                return
            timeline, bar, status, placeholder = [], st.progress(0.0), st.empty(), st.empty()
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            proc = 0
            with st.spinner("Analyzing..."):
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    count += 1
                    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if count % 5 == 0:
                        proc += 1
                        processed, results = detect_emotions_in_frame(frame, st.session_state.model, st.session_state.face_cascade)
                        for res in results:
                            timeline.append({'frame': int(pos), 'time': pos/fps, 'emotion': res['emotion'], 'conf': res['conf']})
                        if proc % 6 == 0:
                            placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption=f"Frame {int(pos)}")
                    if total > 0:
                        bar.progress(min(pos/total, 1.0))
                    status.text(f"Frame {int(pos)}/{total}")
            cap.release(); bar.progress(1.0)
            status.text(f"Done. {proc} processed frames.")
            if timeline:
                df = pd.DataFrame(timeline)
                fig = px.scatter(df, x='time', y='emotion', color='conf', size='conf', title='Video Emotions Timeline', hover_data=['frame','conf'])
                st.plotly_chart(fig, use_container_width=True)
                counts = df['emotion'].value_counts()
                fig2 = px.pie(values=counts.values, names=counts.index, title='Emotion Distribution')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No emotions detected.")
            if os.path.exists(video_file_path):
                try: os.unlink(video_file_path)
                except Exception: pass

def generate_sample_metrics():
    """Generate sample metrics if real ones don't exist"""
    # This function remains the same as your provided version
    if not os.path.exists('performance_metrics.json'):
        sample_metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_name': 'Sample Validation Set', # Added for consistency
            'overall_accuracy': 0.847,
            'precision_macro': 0.832,
            'recall_macro': 0.819,
            'f1_macro': 0.825,
            'roc_auc_macro': 0.891,
            'avg_inference_time': 0.0234,
            'per_class_metrics': {
                'angry': {'precision': 0.78, 'recall': 0.82, 'f1_score': 0.80, 'accuracy': 0.89, 'support': 958},
                'disgust': {'precision': 0.85, 'recall': 0.77, 'f1_score': 0.81, 'accuracy': 0.92, 'support': 111},
                'fear': {'precision': 0.81, 'recall': 0.79, 'f1_score': 0.80, 'accuracy': 0.88, 'support': 1024},
                'happy': {'precision': 0.91, 'recall': 0.89, 'f1_score': 0.90, 'accuracy': 0.94, 'support': 1774},
                'neutral': {'precision': 0.82, 'recall': 0.85, 'f1_score': 0.83, 'accuracy': 0.87, 'support': 1233},
                'sad': {'precision': 0.84, 'recall': 0.81, 'f1_score': 0.82, 'accuracy': 0.90, 'support': 1247},
                'surprise': {'precision': 0.88, 'recall': 0.91, 'f1_score': 0.89, 'accuracy': 0.93, 'support': 831}
            }
        }
        with open('performance_metrics.json', 'w') as f:
            json.dump(sample_metrics, f, indent=2)

def page_metrics():
    st.header("📊 Model Performance Dashboard")
    acc_report = os.path.join(BASE_DIR, "evaluation_accuracy_report.txt")
    perf_json = os.path.join(BASE_DIR, "performance_metrics.json")
    hybrid_json = os.path.join(BASE_DIR, "hybrid_learning_metrics.json")
    hist_png = os.path.join(BASE_DIR, "training_performance_plots.png")  # <-- Use PNG instead of CSV
    conf_png = os.path.join(BASE_DIR, "evaluation_normalized_confusion_matrix.png")
    # --- Classification report ---
    st.subheader("Evaluation Report")
    if os.path.exists(acc_report):
        with open(acc_report, 'r') as f: st.code(f.read())
    else:
        st.info("No accuracy/classification report found.")
    # --- Performance metrics JSON ---
    st.subheader("Performance Metrics")
    if os.path.exists(perf_json):
        with open(perf_json, 'r') as f: metrics = json.load(f)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{metrics.get('overall_accuracy', 0):.2%}")
            st.metric("Macro F1-Score", f"{metrics.get('f1_macro', 0):.3f}")
        with col2:
            st.metric("ROC AUC (Macro)", f"{metrics.get('roc_auc_macro', 0):.3f}")
            st.metric("Avg Inference Time (s)", f"{metrics.get('avg_inference_time', 0):.4f}")
        st.subheader("📈 Per-Class Performance")
        per_class_data = []
        for emotion, emotion_metrics in metrics.get('per_class_metrics', {}).items():
            per_class_data.append({
                'Emotion': emotion.title(),
                'Precision': emotion_metrics.get('precision', 0),
                'Recall': emotion_metrics.get('recall', 0),
                'F1-Score': emotion_metrics.get('f1_score', 0),
                'Support': emotion_metrics.get('support', 0)
            })

        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            st.dataframe(df_per_class, use_container_width=True)
            fig_per_class = px.bar(df_per_class, x='Emotion', y=['Precision', 'Recall', 'F1-Score'], 
                                title='Per-Class Performance Metrics', barmode='group')
            st.plotly_chart(fig_per_class, use_container_width=True)
        else:
            st.info("Per-class metrics data not available in the file.")
        
    # --- Hybrid metrics JSON ---
    st.subheader("Hybrid/Cross-Validation Metrics")
    if os.path.exists(hybrid_json):
        with open(hybrid_json, 'r') as f: hmetrics = json.load(f)
        st.write(hmetrics)
    # --- Training performance plot image ---
    st.subheader("Training Performance Plots")
    if os.path.exists(hist_png):
        st.image(hist_png, use_column_width=True)
    else:
        st.info("No `training_performance_plots.png` found.")
    # --- Confusion Matrix ---
    st.subheader("Normalized Confusion Matrix")
    if os.path.exists(conf_png):
        st.image(conf_png, use_column_width=True)
    else:
        st.info("No confusion matrix image found.")
def page_train():
    st.header("🎓 Model Training & Management")
    st.markdown(f"**Dataset expected at:** `{FER_DATA_DIR}` (`train` and `valid` subfolders)")
    st.markdown(f"**Model output:** `{MODEL_PATH}`")
    st.info("You can run the training script (`emotiondriving.py`) from the app. This may take several minutes and Streamlit may appear unresponsive.")
    if st.button("Start Training Now"):
        with st.spinner("Training script running..."):
            script = os.path.join(BASE_DIR, "emotiondriving.py")
            if not os.path.exists(script):
                st.error("`emotiondriving.py` not found.")
            else:
                try:
                    proc = subprocess.Popen([sys.executable, script], cwd=BASE_DIR)
                    st.success(f"Training started in background (PID {proc.pid}). Monitor console or check output files for progress.")
                except Exception as e:
                    st.error(f"Failed to start training: {e}")
    st.markdown("---")
    st.markdown("After training, reload this page or go to Metrics page to see updated results.")
    st.markdown("You can also run: `python emotiondriving.py` in a terminal.")

def page_about():
    st.header("ℹ️ About This System")
    st.markdown("""
    This dashboard demonstrates deep learning-based facial emotion recognition.
    - **Trained Model:** CNN (Keras/TensorFlow) on FER2013, 7 emotions
    - **Detection:** Haar Cascades
    - **UI:** Streamlit dashboard, Streamlit-WebRTC for live camera
    - **Train/test:** Use the buttons or run scripts directly
    - **Data:** Place FER2013 in `C:\\Users\\HP\\Desktop\\FER2013` with `train`/`valid`
    - **Code & Model:** In `C:\\Users\\HP\\Desktop\\desk`
    ---
    **To use:**
    1. Train new model or use existing.
    2. Try webcam, image, or video detection.
    3. Check metrics and retrain as needed.
    ---
    **Author:** Your Name  
    **Last updated:** {0}
    """.format(time.strftime('%B %Y')))

# --- MAIN APP ---
def main():
    load_core_system()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Real-time Detection", "Image Upload", "Video Upload", "Performance Metrics", "Model Training", "About"
    ])
    if page == "Real-time Detection":
        page_realtime()
    elif page == "Image Upload":
        page_img_upload()
    elif page == "Video Upload":
        page_video_upload()
    elif page == "Performance Metrics":
        page_metrics()
    elif page == "Model Training":
        page_train()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
