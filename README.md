## Vacuum and Spool Detection System - Application Document
### Overview
The Vacuum and Spool Detection System is a real-time computer vision application that uses YOLO object detection to monitor RTSP camera streams for specific object interaction conditions. The system detects and validates overlapping conditions between vacuum/normal, vacuum/suspected, and spool/grove objects, saving evidence when conditions persist for a configured validation period.

## Key Features
### 1. Multi-Camera Support
* Single Camera Mode: Continuous monitoring of one camera stream
* Multi-Camera Mode: Automatic rotation between multiple cameras with configurable time intervals
* RTSP Stream Support: Compatible with various RTSP camera sources
### 2. Intelligent Detection & Validation
* YOLO Object Detection: Real-time object detection using Ultralytics YOLO
* Three Detection Conditions:
1. Vacuum inside/overlapping Normal area
2. Vacuum inside/overlapping Suspected area
3. Spool inside/overlapping Grove area
* Temporal Validation: Conditions must persist for configurable time (default: 3 seconds) before triggering saves
* Intersection-over-Union (IoU): Advanced overlap detection algorithm

### 3. Evidence Collection
* Dual Image Saving: Saves both original and annotated detection frames
* Timestamped Filenaming: Files include precise timestamps with milliseconds
* Organized Storage: Automatic directory creation for output files

### 4. Real-time Monitoring
* Live Display: Real-time video feed with detection overlays
* Performance Metrics: FPS counter and system status display
* Configurable Parameters: Adjustable confidence thresholds and detection settings

## System Requirements
## Hardware Requirements
* CPU: Multi-core processor recommended
* GPU: CUDA-capable GPU recommended for better performance
* RAM: 8GB minimum, 16GB recommended
* Storage: SSD recommended for faster file operations
* Network: Stable network connection for RTSP streams

## Software Requirements
* Python: 3.7 or higher
* OpenCV: 4.5.0 or higher
* Ultralytics YOLO: Latest version
* Pandas: Data handling
* NumPy: Numerical operations

## Installation & Setup
### 1. Prerequisites Installation
```
bash
pip install opencv-python pandas ultralytics numpy
```
### 2. Model Preparation
* Place trained YOLO model file (best.pt) in accessible directory
* Ensure model classes match expected detection classes
### 3. Camera Configuration
* Create CSV file with RTSP URLs (one per line, no header)
* Test RTSP connectivity before deployment
### Configuration Parameters
### Core Parameters in Main Block
```
python
MODEL_PATH = r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train3\weights\best.pt"
CSV_FILE = "rtsp_address_cam_220.csv"
CONFIDENCE_THRESHOLD = 0.5
LINE_THICKNESS = 1
VALIDATION_TIME = 3  # seconds
```
### Class Initialization Parameters
* `model_path`: Path to YOLO model weights
* `csv_file`: CSV containing RTSP URLs
* `inference_time`: Camera rotation time (multi-camera mode)
* `confidence_threshold`: Minimum detection confidence (0.0-1.0)
* `line_thickness`: Bounding box thickness
* `validation_time`: Condition persistence time required for saving

### Usage Instructions
### 1. Basic Operation
```
bash
python find_vacuumeNspool_conf_single_cam.py
```
### 2. Runtime Controls
'q' Key: Exit application gracefully
Automatic Mode Detection: Single vs multi-camera mode auto-detected

3. Output Structure
```
vacuume-spool/
├── YYYYMMDD-HHMMSS-MS-O.jpg  # Original frame
└── YYYYMMDD-HHMMSS-MS-R.jpg  # Result frame with detections
```
### Detection Logic
### Condition Validation Flow
Frame Capture: Read frame from RTSP stream

YOLO Inference: Detect objects with confidence filtering

Overlap Detection: Check for specified object interactions

Temporal Tracking: Monitor condition persistence over time

Validation: Save evidence after validation period elapses

Cleanup: Remove expired conditions from tracking

### Object Interaction Rules
Inside Detection: Complete containment of one bounding box within another

Overlap Detection: IoU > 0.1 threshold

Class Combinations: Specific class pairs trigger different conditions

### Class Reference
### DetailedRTSPYOLOInference
### Key Methods
__init__(): Initialize detector with configuration

process_detections(): Core detection and validation logic

process_camera_stream(): Handle individual camera processing

run(): Main execution loop

calculate_iou(): Intersection over Union calculation

is_inside_or_overlapping(): Spatial relationship detection

### Internal Tracking
condition_timestamps: Track when conditions first occurred

validated_conditions: Track validated persistent conditions

current_camera_index: Multi-camera rotation counter

### Performance Considerations
### Optimization Tips
Confidence Threshold: Adjust based on model performance and false positive rate

Validation Time: Increase to reduce false positives, decrease for faster response

Frame Processing: Consider frame skipping for high-resolution streams

Model Selection: Choose appropriate YOLO model size for hardware capabilities

### Troubleshooting
Connection Issues: Verify RTSP URLs and network connectivity

Model Loading: Check model path and compatibility

Memory Management: Monitor RAM usage with multiple high-resolution streams

File Permissions: Ensure write access to output directory

### Customization
### Adding New Detection Conditions
Modify process_detections() method

Add new class combination checks

Implement appropriate tracking logic

Update condition key generation

### Output Customization
Modify generate_filename() for different naming conventions

Extend saving logic for additional data formats

Add database integration for event logging

### Security Notes
Secure storage of RTSP credentials

File system permission management

Network security for camera streams

Regular updates for dependency vulnerabilities

### Maintenance
Regular model retraining for improved accuracy

Monitor storage capacity for saved images

Log rotation and management

Periodic system health checks

This document provides comprehensive guidance for deploying and maintaining the Vacuum and Spool Detection System. For technical support or customization requests, refer to the code comments and Ultralytics documentation.


