import cv2
import pandas as pd
import os
from datetime import datetime
import time
import logging
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class DetailedRTSPYOLOInference:
    def __init__(self, model_path, csv_file, inference_time=60, confidence_threshold=0.5, line_thickness=2, validation_time=3):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.csv_file = csv_file
        self.inference_time = inference_time
        self.confidence_threshold = confidence_threshold
        self.line_thickness = line_thickness
        self.validation_time = validation_time  # New parameter for validation period
        self.current_camera_index = 0
        self.save_dir = "vacuume-spool"
        self.running = True
        
        # Track conditions over time
        self.condition_timestamps = {}  # Store when each condition first occurred
        self.validated_conditions = set()  # Track conditions that have been validated
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Load RTSP addresses
        self.rtsp_addresses = pd.read_csv(csv_file, header=None)[0].tolist()
        
        # Check if there's only one camera
        self.single_camera_mode = len(self.rtsp_addresses) == 1
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.logger.info(f"Loaded {len(self.rtsp_addresses)} cameras")
        if self.single_camera_mode:
            self.logger.info("Single camera mode: No rotation will occur")
        self.logger.info(f"Model classes: {self.model.names}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Line thickness: {self.line_thickness}")
        self.logger.info(f"Validation time: {self.validation_time} seconds")
    
    def generate_filename(self, image_type):
        """Generate filename in required format"""
        now = datetime.now()
        milliseconds = int(now.microsecond / 10000)
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{milliseconds}-{image_type}.jpg"
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Convert boxes to [x1, y1, x2, y2] format if needed
        if hasattr(box1, 'cpu'):
            box1 = box1.cpu().numpy()
        if hasattr(box2, 'cpu'):
            box2 = box2.cpu().numpy()
            
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    
    def is_inside_or_overlapping(self, inner_box, outer_box, iou_threshold=0.1):
        """
        Check if inner_box is inside or overlapping with outer_box
        Returns True if:
        1. inner_box is completely inside outer_box
        2. inner_box overlaps with outer_box (IoU > threshold)
        """
        # Convert boxes to [x1, y1, x2, y2] format if needed
        if hasattr(inner_box, 'cpu'):
            inner_box = inner_box.cpu().numpy()
        if hasattr(outer_box, 'cpu'):
            outer_box = outer_box.cpu().numpy()
            
        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box
        
        # Check if inner box is completely inside outer box
        if (x1_i >= x1_o and y1_i >= y1_o and 
            x2_i <= x2_o and y2_i <= y2_o):
            return True
        
        # Check for overlap using IoU
        iou = self.calculate_iou(inner_box, outer_box)
        return iou > iou_threshold
    
    def get_condition_key(self, condition_type, det1, det2):
        """Generate a unique key for a condition to track it over time"""
        # Create a key based on the condition type and approximate positions
        # This helps track the same condition across frames
        bbox1 = det1['bbox'].cpu().numpy() if hasattr(det1['bbox'], 'cpu') else det1['bbox']
        bbox2 = det2['bbox'].cpu().numpy() if hasattr(det2['bbox'], 'cpu') else det2['bbox']
        
        # Use rounded coordinates to group similar detections
        key = f"{condition_type}_{int(bbox1[0]/10)}_{int(bbox1[1]/10)}_{int(bbox2[0]/10)}_{int(bbox2[1]/10)}"
        return key
    
    def process_detections(self, result, original_frame):
        """Process detections and save images only if conditions persist for validation_time seconds"""
        if result.boxes is None:
            return
        
        # Organize detections by class (filtered by confidence threshold)
        detections_by_class = {}
        for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            confidence = float(conf)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                continue
                
            class_name = result.names[int(cls)]
            
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            
            detections_by_class[class_name].append({
                'bbox': xyxy,
                'confidence': confidence
            })
        
        # Log detected classes above threshold
        if detections_by_class:
            detected_classes = ", ".join([f"{cls}({len(dets)})" for cls, dets in detections_by_class.items()])
            self.logger.debug(f"Detections above threshold: {detected_classes}")
        
        current_time = time.time()
        current_conditions = set()
        
        # Check for the required overlap conditions
        # Condition 1: vacuume overlapping or in normal
        if 'vacuume' in detections_by_class and 'normal' in detections_by_class:
            for vacuume_det in detections_by_class['vacuume']:
                for normal_det in detections_by_class['normal']:
                    if self.is_inside_or_overlapping(vacuume_det['bbox'], normal_det['bbox']):
                        condition_key = self.get_condition_key("vacuume_normal", vacuume_det, normal_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_timestamps:
                            self.condition_timestamps[condition_key] = current_time
                            self.logger.debug(f"New condition detected: {condition_key}")
                        break
                if any("vacuume_normal" in key for key in current_conditions):
                    break
        
        # Condition 2: vacuume overlapping or in suspected
        if 'vacuume' in detections_by_class and 'suspected' in detections_by_class:
            for vacuume_det in detections_by_class['vacuume']:
                for suspected_det in detections_by_class['suspected']:
                    if self.is_inside_or_overlapping(vacuume_det['bbox'], suspected_det['bbox']):
                        condition_key = self.get_condition_key("vacuume_suspected", vacuume_det, suspected_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_timestamps:
                            self.condition_timestamps[condition_key] = current_time
                            self.logger.debug(f"New condition detected: {condition_key}")
                        break
                if any("vacuume_suspected" in key for key in current_conditions):
                    break
        
        # Condition 3: spool overlapping or in glove
        if 'spool' in detections_by_class and 'grove' in detections_by_class:
            for spool_det in detections_by_class['spool']:
                for grove_det in detections_by_class['grove']:
                    if self.is_inside_or_overlapping(spool_det['bbox'], grove_det['bbox']):
                        condition_key = self.get_condition_key("spool_grove", spool_det, grove_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_timestamps:
                            self.condition_timestamps[condition_key] = current_time
                            self.logger.debug(f"New condition detected: {condition_key}")
                        break
                if any("spool_grove" in key for key in current_conditions):
                    break
        
        # Remove conditions that are no longer present
        expired_conditions = [k for k in self.condition_timestamps.keys() if k not in current_conditions]
        for condition in expired_conditions:
            self.logger.debug(f"Condition expired: {condition}")
            if condition in self.validated_conditions:
                self.validated_conditions.remove(condition)
            del self.condition_timestamps[condition]
        
        # Check for validated conditions (those that have persisted for validation_time)
        should_save = False
        overlap_info = []
        
        for condition_key in current_conditions:
            if condition_key in self.condition_timestamps:
                time_elapsed = current_time - self.condition_timestamps[condition_key]
                
                if time_elapsed >= self.validation_time and condition_key not in self.validated_conditions:
                    # Condition has been valid for the required time
                    self.validated_conditions.add(condition_key)
                    should_save = True
                    
                    # Extract condition type for logging
                    if "vacuume_normal" in condition_key:
                        overlap_info.append(f"vacuume in/overlapping normal (validated for {time_elapsed:.1f}s)")
                    elif "vacuume_suspected" in condition_key:
                        overlap_info.append(f"vacuume in/overlapping suspected (validated for {time_elapsed:.1f}s)")
                    elif "spool_grove" in condition_key:
                        overlap_info.append(f"spool in/overlapping grove (validated for {time_elapsed:.1f}s)")
        
        # Save images if any validated condition is met
        if should_save:
            # Save original frame
            original_filename = self.generate_filename("O")
            cv2.imwrite(os.path.join(self.save_dir, original_filename), original_frame)
            
            # Save result frame with overlay
            result_frame = result.plot(line_width=self.line_thickness)
            result_filename = self.generate_filename("R")
            cv2.imwrite(os.path.join(self.save_dir, result_filename), result_frame)
            
            self.logger.info(f"Detection saved: {original_filename}, {result_filename}")
            for info in overlap_info:
                self.logger.info(f"  - Validated condition: {info}")
    
    def process_camera_stream(self, rtsp_url, duration, single_camera_mode=False):
        """Process a single camera stream"""
        self.logger.info(f"Processing: {rtsp_url}")
        
        # Reset condition tracking when starting a new camera
        self.condition_timestamps = {}
        self.validated_conditions = set()
        
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to connect: {rtsp_url}")
            return
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.running:
                # For single camera mode, we run indefinitely
                # For multi-camera mode, we check the duration
                if not single_camera_mode and (time.time() - start_time) >= duration:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # YOLO inference
                results = self.model(frame, verbose=False)
                result = results[0]
                
                # Process detections
                self.process_detections(result, frame)
                
                # Display
                display_frame = result.plot(line_width=self.line_thickness)
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add info overlay
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                # Handle time display differently for single vs multi camera
                if single_camera_mode:
                    cv2.putText(display_frame, f"Elapsed: {int(elapsed)}s", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    cv2.putText(display_frame, "Mode: Continuous", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                else:
                    remaining_time = int(duration - elapsed)
                    cv2.putText(display_frame, f"Time: {remaining_time}s", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Camera: {self.current_camera_index + 1}/{len(self.rtsp_addresses)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.putText(display_frame, f"Confidence: {self.confidence_threshold}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Validation: {self.validation_time}s", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.imshow('YOLO RTSP Inference', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            cap.release()

    def run(self):
        """Main execution loop"""
        self.logger.info("Starting camera processing...")
        
        while self.running:
            url = self.rtsp_addresses[self.current_camera_index]
            
            # For single camera mode, run indefinitely
            if self.single_camera_mode:
                self.process_camera_stream(url, self.inference_time, single_camera_mode=True)
            else:
                # For multiple cameras, use the specified inference time and rotate
                self.process_camera_stream(url, self.inference_time, single_camera_mode=False)
                
                if not self.running:
                    break
                    
                self.current_camera_index = (self.current_camera_index + 1) % len(self.rtsp_addresses)
                self.logger.info(f"Rotating to camera {self.current_camera_index + 1}")
        
        cv2.destroyAllWindows()

# Modified main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    MODEL_PATH = r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train3\weights\best.pt"
    CSV_FILE = "rtsp_address_cam_220.csv"#"rtsp_address.csv" #
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold
    LINE_THICKNESS = 1  # Parameter for bounding box thickness
    VALIDATION_TIME = 3  # New parameter: seconds to validate conditions before saving
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CSV_FILE):
        print("Error: Model or CSV file not found")
        exit(1)
    
    detector = DetailedRTSPYOLOInference(
        MODEL_PATH, 
        CSV_FILE, 
        confidence_threshold=CONFIDENCE_THRESHOLD,
        line_thickness=LINE_THICKNESS,
        validation_time=VALIDATION_TIME  # Pass the validation time
    )
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cv2.destroyAllWindows()