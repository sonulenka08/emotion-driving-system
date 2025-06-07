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
import plotly.graph_objects as go

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
st.set_page_config(
    page_title="AI based Emotion Recognition System for enhanced driving safety",
    page_icon="😊",
    layout="wide"
)

st.title("🎭 AI based Emotion Recognition System for enhanced driving safety")

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

def page_model_structure():
    st.header("🏗️ Model Architecture")
    
    if not st.session_state.system_ready:
        st.warning("Model not loaded. Please load the system components first.")
        if st.button("Load System Components"):
            load_core_system()
            st.rerun()
        return
    
    if st.session_state.model is None:
        st.error("No model available to display structure.")
        return
    
    try:
        # Model Summary
        st.subheader("📋 Model Summary")
        
        # Capture model summary as string
        import io
        from contextlib import redirect_stdout
        
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            st.session_state.model.summary()
        summary_string = summary_buffer.getvalue()
        
        # Display in code block for better formatting
        st.code(summary_string, language='text')
        
        # Model Configuration
        st.subheader("⚙️ Model Configuration")
        
        # Get model config
        model_config = st.session_state.model.get_config()
        
        # Display basic model info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_config.get('name', 'Sequential'))
            st.metric("Total Layers", len(st.session_state.model.layers))
        
        with col2:
            # Count parameters
            total_params = st.session_state.model.count_params()
            trainable_params = sum([layer.count_params() for layer in st.session_state.model.layers if layer.trainable])
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        
        with col3:
            # Input and output shapes - handle safely
            try:
                if hasattr(st.session_state.model, 'input_shape'):
                    input_shape = str(st.session_state.model.input_shape)
                elif hasattr(st.session_state.model, 'input') and hasattr(st.session_state.model.input, 'shape'):
                    input_shape = str(st.session_state.model.input.shape)
                else:
                    input_shape = "N/A"
            except:
                input_shape = "N/A"
                
            try:
                if hasattr(st.session_state.model, 'output_shape'):
                    output_shape = str(st.session_state.model.output_shape)
                elif hasattr(st.session_state.model, 'output') and hasattr(st.session_state.model.output, 'shape'):
                    output_shape = str(st.session_state.model.output.shape)
                else:
                    output_shape = "N/A"
            except:
                output_shape = "N/A"
                
            st.metric("Input Shape", input_shape)
            st.metric("Output Shape", output_shape)
        
        # Layer Details
        st.subheader("🔍 Layer Details")
        
        layer_data = []
        for i, layer in enumerate(st.session_state.model.layers):
            # Get output shape safely
            try:
                if hasattr(layer, 'output_shape'):
                    output_shape = str(layer.output_shape)
                elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                    output_shape = str(layer.output.shape)
                else:
                    output_shape = "N/A"
            except:
                output_shape = "N/A"
            
            layer_info = {
                'Layer #': i + 1,
                'Layer Name': layer.name,
                'Layer Type': layer.__class__.__name__,
                'Output Shape': output_shape,
                'Parameters': layer.count_params(),
                'Trainable': layer.trainable
            }
            
            # Add specific layer configurations
            if hasattr(layer, 'filters'):
                layer_info['Filters'] = layer.filters
            if hasattr(layer, 'kernel_size'):
                layer_info['Kernel Size'] = str(layer.kernel_size)
            if hasattr(layer, 'strides'):
                layer_info['Strides'] = str(layer.strides)
            if hasattr(layer, 'activation'):
                if hasattr(layer.activation, '__name__'):
                    layer_info['Activation'] = layer.activation.__name__
                else:
                    layer_info['Activation'] = str(layer.activation)
            if hasattr(layer, 'units'):
                layer_info['Units'] = layer.units
            if hasattr(layer, 'rate'):
                layer_info['Dropout Rate'] = layer.rate
            if hasattr(layer, 'pool_size'):
                layer_info['Pool Size'] = str(layer.pool_size)
                
            layer_data.append(layer_info)
        
        # Display layer table
        layer_df = pd.DataFrame(layer_data)
        st.dataframe(layer_df, use_container_width=True)
        
        # Model Visualization
        st.subheader("📊 Parameter Distribution")
        
        # Create parameter distribution chart
        param_data = []
        for layer in st.session_state.model.layers:
            if layer.count_params() > 0:
                param_data.append({
                    'Layer': f"{layer.name} ({layer.__class__.__name__})",
                    'Parameters': layer.count_params(),
                    'Trainable': 'Trainable' if layer.trainable else 'Non-trainable'
                })
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            
            # Bar chart of parameters per layer
            fig_params = px.bar(
                param_df, 
                x='Layer', 
                y='Parameters',
                color='Trainable',
                title='Parameters per Layer',
                text='Parameters'
            )
            fig_params.update_layout(
                xaxis_title="Layer",
                yaxis_title="Number of Parameters",
                xaxis={'tickangle': 45},
                height=500
            )
            fig_params.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_params, use_container_width=True)
            
            # Pie chart of parameter distribution
            fig_pie = px.pie(
                param_df,
                values='Parameters',
                names='Layer',
                title='Parameter Distribution Across Layers'
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Model Compilation Details
        st.subheader("🔧 Compilation Details")
        
        try:
            # Get optimizer details
            optimizer = st.session_state.model.optimizer
            if optimizer:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Optimizer:**")
                    st.write(f"- Type: {optimizer.__class__.__name__}")
                    if hasattr(optimizer, 'learning_rate'):
                        lr = optimizer.learning_rate
                        if hasattr(lr, 'numpy'):
                            st.write(f"- Learning Rate: {lr.numpy()}")
                        else:
                            st.write(f"- Learning Rate: {lr}")
                
                with col2:
                    st.write("**Loss Function:**")
                    if hasattr(st.session_state.model, 'loss'):
                        st.write(f"- {st.session_state.model.loss}")
                    
                    st.write("**Metrics:**")
                    if hasattr(st.session_state.model, 'metrics_names'):
                        for metric in st.session_state.model.metrics_names:
                            st.write(f"- {metric}")
        except Exception as e:
            st.info("Compilation details not fully available.")
        
        # Download model architecture
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download model summary as text
            if st.button("📄 Download Model Summary"):
                st.download_button(
                    label="Download Summary (TXT)",
                    data=summary_string,
                    file_name="model_summary.txt",
                    mime="text/plain"
                )
        
        with col2:
            # Download model architecture as JSON
            if st.button("📋 Download Architecture"):
                architecture_json = st.session_state.model.to_json(indent=2)
                st.download_button(
                    label="Download Architecture (JSON)",
                    data=architecture_json,
                    file_name="model_architecture.json",
                    mime="application/json"
                )
        
    except Exception as e:
        st.error(f"Error displaying model structure: {str(e)}")
        st.info("Make sure the model is properly loaded and accessible.")


def page_metrics():
    st.header("📊 Model Performance Dashboard")
    acc_report = os.path.join(BASE_DIR, "evaluation_accuracy_report.txt")
    perf_json = os.path.join(BASE_DIR, "performance_metrics.json")
    hybrid_json = os.path.join(BASE_DIR, "hybrid_learning_metrics.json")
    hist_png = os.path.join(BASE_DIR, "training_performance_plots.png")
    conf_png = os.path.join(BASE_DIR, "evaluation_normalized_confusion_matrix.png")
    
    # Required imports (add these at the top of your file)
    import numpy as np
    
    # --- Classification report ---
    st.subheader("Evaluation Report")
    if os.path.exists(acc_report):
        with open(acc_report, 'r') as f: 
            st.code(f.read())
    else:
        st.info("No accuracy/classification report found.")
    
    # --- Performance metrics JSON ---
    st.subheader("Performance Metrics")
    if os.path.exists(perf_json):
        with open(perf_json, 'r') as f: 
            metrics = json.load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{metrics.get('overall_accuracy', 0):.2%}")
            st.metric("Macro F1-Score", f"{metrics.get('f1_macro', 0):.3f}")
        with col2:
            st.metric("ROC AUC (Macro)", f"{metrics.get('roc_auc_macro', 0):.3f}")
            st.metric("Avg Inference Time (s)", f"{metrics.get('avg_inference_time', 0):.4f}")
        
        # Add macro precision and recall metrics
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Macro Precision", f"{metrics.get('precision_macro', 0):.3f}")
        with col4:
            st.metric("Macro Recall", f"{metrics.get('recall_macro', 0):.3f}")
        
        st.subheader("📈 Per-Class Performance")
        per_class_data = []
        
        # Get per_class_metrics and process each emotion
        per_class_metrics = metrics.get('per_class_metrics', {})
        
        for emotion_key, emotion_metrics in per_class_metrics.items():
            per_class_data.append({
                'Emotion': emotion_key.title(),
                'Precision': float(emotion_metrics.get('precision', 0)),
                'Recall': float(emotion_metrics.get('recall', 0)),
                'F1-Score': float(emotion_metrics.get('f1_score', 0)),
                'Accuracy': float(emotion_metrics.get('accuracy', 0)),
                'Support': int(emotion_metrics.get('support', 0))
            })

        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            
            # Display the dataframe
            st.dataframe(df_per_class, use_container_width=True)
            
            # Create the chart using a more reliable approach
            try:
                # Method 1: Create chart with explicit data preparation
                emotions = df_per_class['Emotion'].tolist()
                precision_vals = df_per_class['Precision'].tolist()
                recall_vals = df_per_class['Recall'].tolist()
                f1_vals = df_per_class['F1-Score'].tolist()
                accuracy_vals = df_per_class['Accuracy'].tolist()
                
                # Create the figure manually
                fig_per_class = go.Figure()
                
                # Add bars for each metric
                fig_per_class.add_trace(go.Bar(
                    name='Precision',
                    x=emotions,
                    y=precision_vals,
                    marker_color='#1f77b4',
                    text=[f'{val:.3f}' for val in precision_vals],
                    textposition='outside'
                ))
                
                fig_per_class.add_trace(go.Bar(
                    name='Recall',
                    x=emotions,
                    y=recall_vals,
                    marker_color='#ff7f0e',
                    text=[f'{val:.3f}' for val in recall_vals],
                    textposition='outside'
                ))
                
                fig_per_class.add_trace(go.Bar(
                    name='F1-Score',
                    x=emotions,
                    y=f1_vals,
                    marker_color='#2ca02c',
                    text=[f'{val:.3f}' for val in f1_vals],
                    textposition='outside'
                ))
                
                fig_per_class.add_trace(go.Bar(
                    name='Accuracy',
                    x=emotions,
                    y=accuracy_vals,
                    marker_color='#d62728',
                    text=[f'{val:.3f}' for val in accuracy_vals],
                    textposition='outside'
                ))
                
                # Update layout
                fig_per_class.update_layout(
                    title='Per-Class Performance Metrics',
                    xaxis_title="Emotion",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    barmode='group',
                    legend_title="Metrics",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_per_class, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                
                # Fallback: Try with simpler plotly express approach
                try:
                    st.write("Trying alternative chart method...")
                    
                    # Ensure all numeric columns are properly typed
                    df_chart = df_per_class.copy()
                    numeric_cols = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
                    for col in numeric_cols:
                        df_chart[col] = pd.to_numeric(df_chart[col], errors='coerce')
                    
                    # Create melted dataframe
                    df_melted = df_chart.melt(
                        id_vars=['Emotion'], 
                        value_vars=numeric_cols,
                        var_name='Metric', 
                        value_name='Score'
                    )
                    
                    # Ensure Score is numeric
                    df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
                    
                    # Remove any NaN values
                    df_melted = df_melted.dropna()
                    
                    if not df_melted.empty:
                        fig_alt = px.bar(
                            df_melted, 
                            x='Emotion', 
                            y='Score', 
                            color='Metric',
                            title='Per-Class Performance Metrics (Alternative)',
                            barmode='group',
                            range_y=[0, 1],
                            height=500
                        )
                        
                        st.plotly_chart(fig_alt, use_container_width=True)
                    else:
                        st.error("No valid data for plotting after cleaning")
                        
                except Exception as e2:
                    st.error(f"Alternative chart method also failed: {str(e2)}")
                    
                    # Final fallback: Show raw data and simple metrics
                    st.write("Chart creation failed. Here's the summary data:")
                    
                    # Create simple metric display
                    for _, row in df_per_class.iterrows():
                        st.write(f"**{row['Emotion']}:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Precision", f"{row['Precision']:.3f}")
                        with col2:
                            st.metric("Recall", f"{row['Recall']:.3f}")
                        with col3:
                            st.metric("F1-Score", f"{row['F1-Score']:.3f}")
                        with col4:
                            st.metric("Accuracy", f"{row['Accuracy']:.3f}")
                        st.write("---")
            
        else:
            st.info("Per-class metrics data not available in the file.")
        
    else:
        st.info("No performance metrics JSON file found.")
        
    # --- Hybrid metrics JSON ---
    st.subheader("🔄 Hybrid/Cross-Validation Metrics")
    if os.path.exists(hybrid_json):
        with open(hybrid_json, 'r') as f: 
            hmetrics = json.load(f)
        
        # Create interactive tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 CV Scores", "🎯 Fold Details", "📋 Raw Data"])
        
        with tab1:
            st.write("### Cross-Validation Summary")
            
            # Extract key metrics if available
            if 'cv_scores' in hmetrics:
                cv_scores = hmetrics['cv_scores']
                if isinstance(cv_scores, list) and len(cv_scores) > 0:
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean CV Score", f"{cv_mean:.4f}")
                    with col2:
                        st.metric("Std Deviation", f"{cv_std:.4f}")
                    with col3:
                        st.metric("Best Fold", f"{max(cv_scores):.4f}")
                    with col4:
                        st.metric("Worst Fold", f"{min(cv_scores):.4f}")
            
            # Display other summary metrics
            summary_metrics = ['mean_accuracy', 'std_accuracy', 'mean_f1', 'std_f1', 'mean_precision', 'std_precision', 'mean_recall', 'std_recall']
            
            if any(metric in hmetrics for metric in summary_metrics):
                st.write("### Performance Summary")
                
                summary_cols = st.columns(2)
                with summary_cols[0]:
                    if 'mean_accuracy' in hmetrics:
                        st.metric("Mean Accuracy", f"{hmetrics['mean_accuracy']:.4f}", 
                        delta=f"±{hmetrics.get('std_accuracy', 0):.4f}")
                    if 'mean_f1' in hmetrics:
                        st.metric("Mean F1-Score", f"{hmetrics['mean_f1']:.4f}", 
                        delta=f"±{hmetrics.get('std_f1', 0):.4f}")
                
                with summary_cols[1]:
                    if 'mean_precision' in hmetrics:
                        st.metric("Mean Precision", f"{hmetrics['mean_precision']:.4f}", 
                        delta=f"±{hmetrics.get('std_precision', 0):.4f}")
                    if 'mean_recall' in hmetrics:
                        st.metric("Mean Recall", f"{hmetrics['mean_recall']:.4f}", 
                        delta=f"±{hmetrics.get('std_recall', 0):.4f}")
        
        with tab2:
            st.write("### Cross-Validation Scores Visualization")
            
            # Plot CV scores if available
            if 'cv_scores' in hmetrics and isinstance(hmetrics['cv_scores'], list):
                cv_scores = hmetrics['cv_scores']
                
                # Create fold-wise performance chart
                fold_data = pd.DataFrame({
                    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                    'Score': cv_scores
                })
                
                # Line chart showing CV scores across folds
                fig_cv = px.line(fold_data, x='Fold', y='Score', 
                    title='Cross-Validation Scores by Fold',
                    markers=True, line_shape='linear')
                
                # Add mean line
                mean_score = np.mean(cv_scores)
                fig_cv.add_hline(y=mean_score, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_score:.4f}")
                
                fig_cv.update_layout(height=400, yaxis_title="Score", xaxis_title="Fold")
                st.plotly_chart(fig_cv, use_container_width=True)
                
                # Box plot for score distribution
                fig_box = px.box(y=cv_scores, title='Cross-Validation Score Distribution')
                fig_box.update_layout(height=300, yaxis_title="Score")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Plot other metrics if available
            metric_keys = [key for key in hmetrics.keys() if 'scores' in key.lower() or 'folds' in key.lower()]
            
            if metric_keys:
                st.write("### Other Cross-Validation Metrics")
                
                selected_metric = st.selectbox("Select metric to visualize:", metric_keys)
                
                if selected_metric in hmetrics:
                    metric_data = hmetrics[selected_metric]
                    if isinstance(metric_data, list):
                        metric_df = pd.DataFrame({
                            'Fold': [f'Fold {i+1}' for i in range(len(metric_data))],
                            'Value': metric_data
                        })
                        
                        fig_metric = px.bar(metric_df, x='Fold', y='Value',
                        title=f'{selected_metric.replace("_", " ").title()} by Fold')
                        fig_metric.update_layout(height=400)
                        st.plotly_chart(fig_metric, use_container_width=True)
        
        with tab3:
            st.write("### Detailed Fold Analysis")
            
            # Create detailed fold comparison
            fold_metrics = {}
            
            # Look for fold-specific data
            for key, value in hmetrics.items():
                if isinstance(value, list) and len(value) > 1:
                    fold_metrics[key] = value
            
            if fold_metrics:
                # Create comparison table
                max_folds = max(len(values) for values in fold_metrics.values())
                
                fold_comparison_data = []
                for i in range(max_folds):
                    row = {'Fold': f'Fold {i+1}'}
                    for metric_name, metric_values in fold_metrics.items():
                        if i < len(metric_values):
                            row[metric_name.replace('_', ' ').title()] = metric_values[i]
                    fold_comparison_data.append(row)
                
                fold_df = pd.DataFrame(fold_comparison_data)
                
                # Interactive data table
                st.write("#### Fold-by-Fold Comparison")
                st.dataframe(fold_df, use_container_width=True)
                
                # Heatmap of fold performance
                if len(fold_df.columns) > 2:  # More than just Fold column
                    numeric_columns = [col for col in fold_df.columns if col != 'Fold']
                    
                    # Prepare data for heatmap
                    heatmap_data = fold_df.set_index('Fold')[numeric_columns].T
                    
                    fig_heatmap = px.imshow(heatmap_data, 
                        title='Cross-Validation Performance Heatmap',
                        labels=dict(x="Fold", y="Metric", color="Score"),
                        aspect="auto")
                    fig_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Statistical analysis
                st.write("#### Statistical Analysis")
                
                stats_data = []
                for metric_name, metric_values in fold_metrics.items():
                    if len(metric_values) > 1:
                        stats_data.append({
                            'Metric': metric_name.replace('_', ' ').title(),
                            'Mean': np.mean(metric_values),
                            'Std': np.std(metric_values),
                            'Min': np.min(metric_values),
                            'Max': np.max(metric_values),
                            'Range': np.max(metric_values) - np.min(metric_values)
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No fold-specific metrics found for detailed analysis.")
        
        with tab4:
            st.write("### Raw JSON Data")
            
            # Expandable sections for different parts of the data
            if isinstance(hmetrics, dict):
                for section_name, section_data in hmetrics.items():
                    with st.expander(f"📁 {section_name.replace('_', ' ').title()}"):
                        if isinstance(section_data, (dict, list)):
                            st.json(section_data)
                        else:
                            st.write(section_data)
            
            # Download button for raw data
            st.download_button(
                label="📥 Download Raw Metrics (JSON)",
                data=json.dumps(hmetrics, indent=2),
                file_name="hybrid_learning_metrics.json",
                mime="application/json"
            )
            
            # Search functionality
            st.write("#### Search in Raw Data")
            search_term = st.text_input("Search for specific metrics or values:")
            
            if search_term:
                search_results = []
                
                def search_nested_dict(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            if search_term.lower() in key.lower():
                                search_results.append({"Path": current_path, "Key": key, "Value": value})
                            search_nested_dict(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            current_path = f"{path}[{i}]"
                            search_nested_dict(item, current_path)
                    else:
                        if search_term.lower() in str(obj).lower():
                            search_results.append({"Path": path, "Key": "Value", "Value": obj})
                
                search_nested_dict(hmetrics)
                
                if search_results:
                    st.write(f"Found {len(search_results)} matches:")
                    search_df = pd.DataFrame(search_results)
                    st.dataframe(search_df, use_container_width=True)
                else:
                    st.write("No matches found.")
        
    else:
        st.info("No hybrid learning metrics found.")
    
        
    # --- Training performance plot image ---
    st.subheader("Training Performance Plots")
    if os.path.exists(hist_png):
        try:
            # Use PIL to open the image first, then display
            image = Image.open(hist_png)
            st.image(image, caption="Training Performance", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading training performance plot: {e}")
    else:
        st.warning("Training performance plot not found.")

    # --- Confusion Matrix ---
    st.subheader("Normalized Confusion Matrix")
    if os.path.exists(conf_png):
        try:
            # Use PIL to open the image first, then display
            image = Image.open(conf_png)
            st.image(image, caption="Confusion Matrix", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading confusion matrix: {e}")
    else:
        st.warning("Confusion matrix not found.")

def page_about():
    st.header("ℹ️ About Section")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎭 Advanced Emotion Recognition System
        
        This system uses state-of-the-art deep learning techniques to detect and classify human emotions
        from facial expressions in real-time with high accuracy.
        
        ### ✨ Key Features:
        - **Real-time Detection**: Live emotion analysis via webcam
        - **Image Analysis**: Upload and analyze static images
        - **Video Processing**: Analyze emotions in video files
        - **Comprehensive Metrics**: Detailed performance analysis
        - **Cross-validation**: Robust model evaluation
        - **7 Emotion Classes**: angry, disgust, fear, happy, neutral, sad, surprise
        
        ### 🔧 Technology Stack:
        - **Deep Learning**: TensorFlow/Keras CNN architecture
        - **Computer Vision**: OpenCV for face detection and processing
        - **Web Interface**: Streamlit for interactive dashboard
        - **Real-time Streaming**: WebRTC for live video processing
        - **Visualization**: Plotly for interactive charts
        - **Dataset**: FER2013 facial expression recognition dataset
        
        ### 📊 Model Architecture:
        - **Input**: 48x48 grayscale facial images
        - **Architecture**: Multi-layer CNN with batch normalization
        - **Layers**: 3 convolutional blocks + 2 dense layers
        - **Activation**: ReLU + Softmax output
        - **Optimization**: Adam optimizer with learning rate scheduling
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Performance Stats:
        """)
        
        # Performance summary
        perf_data = {
            'Metric': 'Accuracy',
            'Score': ~76
        }
        
        st.markdown("""
        ### 🚀 Recent Updates:
        - ✅ Added real-time WebRTC streaming
        - ✅ Enhanced cross-validation metrics  
        - ✅ Improved UI/UX design
        - ✅ Added video file analysis
        - ✅ Performance dashboard
        """)
    
    # Technical details
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        #### Model Training Details:
        - **Dataset**: 35,887 images (28,709 training, 7,178 testing)
        - **Preprocessing**: Face detection + normalization + augmentation
        - **Training Time**: ~2-3 hours on GPU
        - **Validation**: 5-fold stratified cross-validation
        - **Early Stopping**: Prevents overfitting
        
        #### System Requirements:
        - **Python**: 3.8+
        - **Memory**: 8GB+ RAM recommended
        - **GPU**: Optional but recommended for training
        - **Webcam**: Required for real-time detection
        """)
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
                
    # Contact/Support
    ---
    ### 💬 Support & Feedback
    
    For technical support, feature requests, or bug reports, please contact the development team.
    
    
    **Author:** Your Name  
    **Last updated:** {0}
    """.format(time.strftime('%B %Y')))

# --- MAIN APP ---
def main():
    load_core_system()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Real-time Detection", "Image Upload", "Video Upload", "Performance Metrics","Model Structure", "Model Training", "About"
    ])
    if page == "Real-time Detection":
        page_realtime()
    elif page == "Image Upload":
        page_img_upload()
    elif page == "Video Upload":
        page_video_upload()
    elif page == "Performance Metrics":
        page_metrics()
    elif page == "Model Structure":
        page_model_structure()
    elif page == "Model Training":
        page_train()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
