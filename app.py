import streamlit as st
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import threading
import logging
from queue import Queue

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the directory and JSON file
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

class ImprovedPersonDetector:
    def __init__(self):
        """Initialize the detector with authorized images"""
        self.authorized_faces = []
        self.authorized_image_paths = []
        self.lock = threading.Lock()
        
        if not os.path.exists(AUTHORIZED_DIR):
            os.makedirs(AUTHORIZED_DIR)
        
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    self.authorized_image_paths = json.load(f)
            except Exception as e:
                logger.error(f"Error loading JSON: {e}")
                self.authorized_image_paths = []
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        try:
            self.mtcnn = MTCNN(keep_all=True, device=self.device, margin=20, min_face_size=60)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            st.error("Failed to initialize detection models. Check logs.")
    
    def load_images(self):
        """Load authorized images"""
        with self.lock:
            self.authorized_faces = []
            
            # Process images in batches for better performance
            batch_size = 4  # Process 4 images at a time
            for i in range(0, len(self.authorized_image_paths), batch_size):
                batch_paths = self.authorized_image_paths[i:i + batch_size]
                batch_faces = []
                
                for idx, image_path in enumerate(batch_paths, start=i):
                    try:
                        logger.debug(f"Loading image: {image_path}")
                        # Load and convert image in one step
                        person_image = Image.open(image_path).convert('RGB')
                        
                        # Process face detection
                        faces = self.mtcnn(person_image)
                        
                        if faces is not None and len(faces) > 0:
                            # Process face embedding
                            face_tensor = faces[0].to(self.device)
                            face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                            
                            batch_faces.append({
                                'id': idx + 1,
                                'image': person_image,
                                'embedding': face_embedding,
                                'name': f"Person_{idx + 1}"
                            })
                            st.sidebar.success(f"✅ Loaded authorized person: Person_{idx + 1}")
                            logger.info(f"Loaded authorized person {idx + 1}")
                        else:
                            st.sidebar.error(f"❌ No face found in authorized image {idx + 1}")
                            logger.warning(f"No face detected in {image_path}")
                    except Exception as e:
                        st.sidebar.error(f"Error loading {image_path}: {str(e)}")
                        logger.error(f"Error loading {image_path}: {e}")
                
                # Add batch to authorized faces
                self.authorized_faces.extend(batch_faces)
                
                # Clear CUDA cache after each batch if using GPU
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            st.sidebar.info(f"Loaded {len(self.authorized_faces)} authorized persons")
    
    def remove_images(self):
        """Remove all authorized images and delete all files listed in JSON"""
        with self.lock:
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, 'r') as f:
                    image_paths = json.load(f)
                
                for file_path in image_paths:
                    if os.path.exists(file_path):
                        try:
                            # Close any open file handles
                            try:
                                with open(file_path, 'rb') as f:
                                    f.close()
                            except:
                                pass
                            
                            # Add a small delay to ensure file is released
                            time.sleep(0.5)
                            
                            # Try to delete the file
                            try:
                                os.remove(file_path)
                                st.sidebar.success(f"🗑️ Deleted file: {os.path.basename(file_path)}")
                            except PermissionError:
                                st.sidebar.error(f"❌ Permission denied: {os.path.basename(file_path)}")
                            except OSError as e:
                                st.sidebar.error(f"❌ Failed to delete {os.path.basename(file_path)}: {str(e)}")
                                # Try alternative deletion method
                                try:
                                    import gc
                                    gc.collect()  # Force garbage collection
                                    time.sleep(0.5)  # Wait for GC
                                    os.remove(file_path)
                                    st.sidebar.success(f"🗑️ Deleted file after retry: {os.path.basename(file_path)}")
                                except:
                                    st.sidebar.error(f"❌ Failed to delete {os.path.basename(file_path)} even after retry")
                        except Exception as e:
                            st.sidebar.error(f"❌ Error handling {os.path.basename(file_path)}: {str(e)}")
                
                # Clear the JSON file
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
            
            self.authorized_faces = []
            self.authorized_image_paths = []
            st.sidebar.success("🗑️ All authorized images removed")
    
    def match_face(self, face_embedding, threshold=0.75):
        """Match a detected face with authorized faces"""
        with self.lock:
            if not self.authorized_faces:
                logger.warning("No authorized faces loaded")
                return None, 0
            
            best_match = None
            best_cosine_score = 0
            best_euclidean_dist = float('inf')
            
            for auth_face in self.authorized_faces:
                cosine_similarity = np.dot(face_embedding, auth_face['embedding'].T) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(auth_face['embedding'])
                )
                cosine_score = (cosine_similarity + 1) / 2
                euclidean_dist = np.linalg.norm(face_embedding - auth_face['embedding'])
                
                if cosine_score > best_cosine_score and euclidean_dist < 1.1:
                    best_cosine_score = cosine_score
                    best_euclidean_dist = euclidean_dist
                    best_match = auth_face
            
            if best_cosine_score >= threshold:
                confidence = float(best_cosine_score * 100)  # Convert to float explicitly
                logger.info(f"Match found: {best_match['name']} with confidence {confidence:.1f}%")
                return best_match, confidence
            return None, 0
        
    def process_frame(self, frame, threshold=0.75):
        """Process frame to detect authorized persons"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            faces = self.mtcnn(pil_image)
            
            detected_persons = []
            
            if faces is not None:
                boxes = self.mtcnn.detect(pil_image)[0]
                logger.debug(f"Detected {len(faces)} faces")
                if boxes is not None:
                    for i, face_tensor in enumerate(faces):
                        if i >= len(boxes):
                            continue
                        
                        face_embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device)).detach().cpu().numpy()
                        match, confidence = self.match_face(face_embedding, threshold)
                        
                        if match:
                            box = boxes[i]
                            left, top, right, bottom = map(int, box)
                            person_name = match['name']
                            detected_persons.append({"name": person_name, "confidence": float(confidence)})
                            
                            # Draw bounding box and info
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            info_text = f"{person_name} ({confidence:.1f}%)"
                            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, info_text, (left + 6, bottom - 6),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                logger.debug("No faces detected in frame")
            
            return frame, detected_persons
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, []

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the authorized_images directory and update JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(AUTHORIZED_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            image_paths = json.load(f)
    else:
        image_paths = []
    
    image_paths.append(file_path)
    with open(JSON_FILE, 'w') as f:
        json.dump(image_paths, f)
    
    return file_path

def test_camera(camera_id, backend):
    """Test if a camera can be opened and read frames"""
    logger.info(f"Testing Camera {camera_id} with backend {backend}")
    
    # Try different backends
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]  # Try CAP_ANY first
    cap = None
    
    for b in backends:
        try:
            time.sleep(0.5)  # Small delay before opening camera
            cap = cv2.VideoCapture(camera_id, b)
            if cap.isOpened():
                logger.info(f"Successfully opened Camera {camera_id} with backend {b}")
                # Try to read a frame immediately
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Successfully read frame from Camera {camera_id}")
                    if cap:
                        cap.release()
                    return True
                else:
                    logger.warning(f"Failed to read frame from Camera {camera_id}")
                    if cap:
                        cap.release()
            else:
                logger.warning(f"Failed to open Camera {camera_id} with backend {b}")
        except Exception as e:
            logger.error(f"Error opening Camera {camera_id} with backend {b}: {str(e)}")
            if cap:
                cap.release()
    
    logger.error(f"Failed to open Camera {camera_id} with any backend")
    return False

def process_camera(detector, camera_id, threshold, stop_event, frame_queue, backend):
    """Process camera feed and push frames to a queue"""
    logger.info(f"Starting Camera {camera_id} with backend {backend}")
    
    # Try different backends
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]  # Try CAP_ANY first
    cap = None
    
    for b in backends:
        try:
            time.sleep(0.5)  # Small delay before opening camera
            cap = cv2.VideoCapture(camera_id, b)
            if cap.isOpened():
                logger.info(f"Successfully opened Camera {camera_id} with backend {b}")
                # Try to read a frame immediately
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Successfully read frame from Camera {camera_id}")
                    break
                else:
                    logger.warning(f"Failed to read frame from Camera {camera_id}")
                    cap.release()
            else:
                logger.warning(f"Failed to open Camera {camera_id} with backend {b}")
        except Exception as e:
            logger.error(f"Error opening Camera {camera_id} with backend {b}: {str(e)}")
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        error_msg = f"Failed to open Camera {camera_id} with any backend"
        logger.error(error_msg)
        frame_queue.put((camera_id, None, error_msg))
        return
    
    try:
        # Set camera properties for smooth scanning
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
        
        logger.info(f"Camera {camera_id} opened successfully")
        frame_count = 0
        last_detection_time = 0  # Track last detection time to avoid spam
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Camera {camera_id} failed to read frame")
                frame_queue.put((camera_id, None, "Frame read failed"))
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.debug(f"Camera {camera_id} processed {frame_count} frames")
            
            # Process frame for detection
            frame, detected_persons = detector.process_frame(frame, threshold)
            
            # Update status text with smooth transitions
            status = f"Camera {camera_id}: " + ("✅ DETECTED" if detected_persons else "Scanning...")
            status_color = (0, 255, 0) if detected_persons else (255, 255, 255)
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Log detections with cooldown
            current_time = time.time()
            if detected_persons and (current_time - last_detection_time) > 1.0:  # 1 second cooldown
                last_detection_time = current_time
                with detector.lock:
                    for person in detected_persons:
                        detection_entry = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Camera": f"Camera {camera_id}",
                            "Person": person["name"],
                            "Confidence": f"{person['confidence']:.1f}%"
                        }
                        # Add detection to queue for main thread to process
                        frame_queue.put((camera_id, None, detection_entry))
                        logger.info(f"Detected {person['name']} on Camera {camera_id}")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put((camera_id, rgb_frame, None))
            
            # Smooth frame rate control
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        error_msg = f"Camera {camera_id} processing error: {str(e)}"
        logger.error(error_msg)
        frame_queue.put((camera_id, None, error_msg))
    finally:
        if cap:
            cap.release()
        logger.info(f"Camera {camera_id} released")

def main():
    st.set_page_config(page_title="Multi-Camera Person Detection", layout="wide")
    
    st.title("Multi-Camera Authorized Person Detection System")
    
    # Initialize all session state variables
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
    
    # Initialize other session state variables if they don't exist
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = {0: False, 1: False}
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    if 'threads' not in st.session_state:
        st.session_state.threads = {0: None, 1: None}
    if 'stop_events' not in st.session_state:
        st.session_state.stop_events = {0: threading.Event(), 1: threading.Event()}
    if 'frame_queues' not in st.session_state:
        st.session_state.frame_queues = {0: Queue(maxsize=1), 1: Queue(maxsize=1)}
    if 'last_frames' not in st.session_state:
        st.session_state.last_frames = {0: None, 1: None}
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    threshold = st.sidebar.slider("Detection Threshold", 0.5, 0.95, 0.75, 0.05)
    backend = st.sidebar.selectbox("Camera Backend", [cv2.CAP_ANY, cv2.CAP_DSHOW], 
                                  format_func=lambda x: "CAP_DSHOW" if x == cv2.CAP_DSHOW else "CAP_ANY")
    
    st.sidebar.subheader("Upload Authorized Persons")
    authorized_uploads = st.sidebar.file_uploader(
        "Upload images of authorized persons", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="auth_uploads"
    )
    
    if st.sidebar.button("Load Authorized Images"):
        if authorized_uploads:
            st.session_state.detector.authorized_image_paths = []
            for uploaded_file in authorized_uploads:
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.detector.authorized_image_paths.append(file_path)
            st.session_state.detector.load_images()
        else:
            st.sidebar.error("Please upload images first!")
    
    if st.sidebar.button("Remove Authorized Images"):
        st.session_state.detector.remove_images()
    
    if st.session_state.detector.authorized_image_paths:
        st.sidebar.subheader("Authorized Persons")
        cols = st.sidebar.columns(min(3, len(st.session_state.detector.authorized_image_paths)))
        for idx, img_path in enumerate(st.session_state.detector.authorized_image_paths):
            with cols[idx % 3]:
                img = Image.open(img_path)
                st.image(img, caption=f"Person {idx + 1}", width=100)
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Live Camera Feeds")
        
        # Global start/stop button
        start_stop_col = st.columns(3)
        with start_stop_col[1]:
            if not any(st.session_state.detection_active.values()):
                if st.button("Start Detection", key="start_both", type="primary"):
                    # First stop any existing threads
                    for cam_id in [0, 1]:
                        if st.session_state.threads[cam_id]:
                            st.session_state.stop_events[cam_id].set()
                            st.session_state.threads[cam_id].join(timeout=1.0)
                            st.session_state.threads[cam_id] = None
                    
                    # Start new threads
                    available_cameras = []
                    for cam_id in [0, 1]:
                        logger.info(f"Attempting to start Camera {cam_id}")
                        if test_camera(cam_id, backend):
                            available_cameras.append(cam_id)
                            st.session_state.detection_active[cam_id] = True
                            st.session_state.stop_events[cam_id].clear()
                            st.session_state.threads[cam_id] = threading.Thread(
                                target=process_camera,
                                args=(st.session_state.detector, cam_id, threshold, 
                                      st.session_state.stop_events[cam_id], 
                                      st.session_state.frame_queues[cam_id], backend)
                            )
                            st.session_state.threads[cam_id].daemon = True
                            st.session_state.threads[cam_id].start()
                            logger.info(f"Started thread for Camera {cam_id}")
                            st.success(f"Camera {cam_id} started successfully")
                            # Add a small delay between starting cameras
                            time.sleep(0.5)
                        else:
                            st.error(f"Failed to start Camera {cam_id}")
                    
                    if not available_cameras:
                        st.error("No cameras available! Check connections and permissions.")
                        logger.error("No cameras detected")
                    else:
                        st.success(f"Started {len(available_cameras)} cameras")
            else:
                if st.button("Stop Detection", key="stop_both", type="secondary"):
                    for cam_id in [0, 1]:
                        if st.session_state.detection_active[cam_id]:
                            st.session_state.detection_active[cam_id] = False
                            st.session_state.stop_events[cam_id].set()
                            if st.session_state.threads[cam_id]:
                                st.session_state.threads[cam_id].join(timeout=1.0)
                            st.session_state.threads[cam_id] = None
                            logger.info(f"Camera {cam_id} thread stopped")
                            st.success(f"Camera {cam_id} stopped")
        
        # Create two columns for cameras
        camera_col1, camera_col2 = st.columns(2)
        
        # Camera 1
        with camera_col1:
            st.subheader("Camera 1")
            camera1_status = st.empty()
            camera1_placeholder = st.empty()
            
            if st.session_state.detection_active[0]:
                camera1_status.success("Camera 1: Active")
                # Display last frame if available
                if st.session_state.last_frames[0] is not None:
                    camera1_placeholder.image(st.session_state.last_frames[0], channels="RGB", use_column_width=True)
            else:
                camera1_status.info("Camera 1: Ready")
        
        # Camera 2
        with camera_col2:
            st.subheader("Camera 2")
            camera2_status = st.empty()
            camera2_placeholder = st.empty()
            
            if st.session_state.detection_active[1]:
                camera2_status.success("Camera 2: Active")
                # Display last frame if available
                if st.session_state.last_frames[1] is not None:
                    camera2_placeholder.image(st.session_state.last_frames[1], channels="RGB", use_column_width=True)
            else:
                camera2_status.info("Camera 2: Ready")
        
        # Update UI with frames from queues
        current_time = time.time()
        update_interval = 0.033  # ~30 FPS
        
        # Check if we should update the UI
        if current_time - st.session_state.last_update >= update_interval:
            for camera_id in [0, 1]:
                if camera_id in st.session_state.frame_queues and not st.session_state.frame_queues[camera_id].empty():
                    cam_id, frame, data = st.session_state.frame_queues[camera_id].get()
                    if frame is not None:
                        # Store the frame
                        st.session_state.last_frames[camera_id] = frame
                        # Display the frame
                        if camera_id == 0:
                            camera1_placeholder.image(frame, channels="RGB", use_column_width=True)
                        else:
                            camera2_placeholder.image(frame, channels="RGB", use_column_width=True)
                    elif isinstance(data, dict):  # Detection entry
                        st.session_state.detection_log.insert(0, data)
                    elif isinstance(data, str):  # Error message
                        if camera_id == 0:
                            camera1_placeholder.error(f"Camera {cam_id}: {data}")
                        else:
                            camera2_placeholder.error(f"Camera {cam_id}: {data}")
            st.session_state.last_update = current_time
        
        # Show overall status
        active_cameras = sum(1 for active in st.session_state.detection_active.values() if active)
        if active_cameras > 0:
            st.success(f"Detection Active (Cameras: {active_cameras}/2)")
        else:
            st.error("Detection Inactive")
        
        # Force UI update
        st.rerun()
    
    with col2:
        st.header("Detection Log")
        with st.container():
            if st.session_state.detection_log:
                # Sort log by timestamp (newest first)
                sorted_log = sorted(st.session_state.detection_log, 
                                  key=lambda x: x["Timestamp"], 
                                  reverse=True)
                st.dataframe(sorted_log, height=400)
            else:
                st.info("No authorized persons detected yet")
        
        if st.button("Refresh UI"):
            st.rerun()

if __name__ == "__main__":
    main()