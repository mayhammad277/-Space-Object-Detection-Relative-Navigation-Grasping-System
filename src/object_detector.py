# src/object_detector.py
"""
Multi-method object detection: YOLOv8, HSV color thresholding, ORB feature matching.
Provides unified interface for detecting spacecraft/debris/objects in orbital imagery.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


class ObjectDetector:
    """Unified object detection supporting multiple computer vision methods."""
    
    def __init__(self, config: Dict):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection configuration dictionary
        """
        self.config = config
        self.method = config['detection']['method']
        self.confidence_threshold = config['detection']['confidence_threshold']
        self.target_class = config['detection'].get('target_class', 'any')
        self.known_width_cm = config['detection']['known_width_cm']
        
        # Color thresholding parameters
        self.hsv_lower = np.array(config['detection'].get('hsv_lower', [20, 100, 100]))
        self.hsv_upper = np.array(config['detection'].get('hsv_upper', [30, 255, 255]))
        
        # Feature matching parameters
        self.reference_image = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        
        # Detection state
        self.target_locked = False
        self.last_known_bbox = None
        self.detection_history: List[bool] = []
        self.max_history = 10
        
        # Initialize selected method
        self._initialize_method()
    
    def _initialize_method(self):
        """Initialize the selected detection method."""
        if self.method == "yolo":
            self._init_yolo()
        elif self.method == "color_threshold":
            self._init_color()
        elif self.method == "feature_match":
            self._init_feature_matcher()
        else:
            print(f"[ObjectDetector] Unknown method '{self.method}', falling back to color")
            self.method = "color_threshold"
            self._init_color()
    
    def _init_yolo(self):
        """Initialize YOLOv8 model."""
        try:
            from ultralytics import YOLO
            model_path = self.config['detection'].get('yolo_model', 'yolov8n.pt')
            self.model = YOLO(model_path)
            print(f"[ObjectDetector] YOLOv8 initialized with {model_path}")
        except ImportError:
            print("[ObjectDetector] ⚠️ ultralytics not installed. Falling back to color detection.")
            print("               Install with: pip install ultralytics")
            self.method = "color_threshold"
            self._init_color()
    
    def _init_color(self):
        """No special initialization needed for color thresholding."""
        print(f"[ObjectDetector] Color thresholding ready (HSV: {self.hsv_lower} - {self.hsv_upper})")
    
    def _init_feature_matcher(self):
        """Initialize ORB feature detector and matcher."""
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Load reference image if provided
        ref_path = self.config['detection'].get('reference_image', None)
        if ref_path:
            self.reference_image = cv2.imread(ref_path)
            if self.reference_image is not None:
                self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
                self.reference_keypoints, self.reference_descriptors = \
                    self.orb.detectAndCompute(self.reference_image, None)
                print(f"[ObjectDetector] Feature matcher ready with reference: {ref_path}")
            else:
                print(f"[ObjectDetector] ⚠️ Could not load reference image: {ref_path}")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]], Optional[float]]:
        """
        Detect target object in frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Tuple containing:
            - success: Whether target was detected
            - bbox: Bounding box [x1, y1, x2, y2] or None
            - centroid: Center point (cx, cy) or None
            - object_width_px: Width of object in pixels for distance calculation
        """
        success, bbox, centroid, obj_width = False, None, None, None
        
        if self.method == "yolo":
            success, bbox, centroid, obj_width = self._detect_yolo(frame)
        elif self.method == "color_threshold":
            success, bbox, centroid, obj_width = self._detect_color(frame)
        elif self.method == "feature_match":
            success, bbox, centroid, obj_width = self._detect_features(frame)
        
        # Update tracking state
        self.detection_history.append(success)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        if success:
            self.target_locked = True
            self.last_known_bbox = bbox
        elif sum(self.detection_history) == 0:
            self.target_locked = False
        
        return success, bbox, centroid, obj_width
    
    def _detect_yolo(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]], Optional[float]]:
        """YOLOv8-based object detection."""
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        best_bbox = None
        best_conf = 0.0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    class_name = self.model.names[cls_id]
                    
                    if conf > best_conf and (self.target_class == "any" or
                                             self.target_class.lower() in class_name.lower()):
                        best_conf = conf
                        best_bbox = boxes.xyxy[i].cpu().numpy()
        
        if best_bbox is not None:
            x1, y1, x2, y2 = best_bbox.astype(int)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            obj_width_px = float(x2 - x1)
            return True, np.array([x1, y1, x2, y2]), centroid, obj_width_px
        
        return False, None, None, None
    
    def _detect_color(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]], Optional[float]]:
        """HSV color thresholding for object detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            min_area = self.config['detection'].get('min_contour_area', 500)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                centroid = (x + w // 2, y + h // 2)
                bbox = np.array([x, y, x + w, y + h])
                return True, bbox, centroid, float(w)
        
        return False, None, None, None
    
    def _detect_features(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]], Optional[float]]:
        """ORB feature matching with reference image."""
        if self.reference_descriptors is None:
            return False, None, None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(des) < 10:
            return False, None, None, None
        
        matches = self.matcher.match(self.reference_descriptors, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:20]
        
        if len(good_matches) >= 10:
            src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt
                                  for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt
                                  for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h, w = self.reference_image.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                x_min = int(min(dst[:, 0, 0]))
                y_min = int(min(dst[:, 0, 1]))
                x_max = int(max(dst[:, 0, 0]))
                y_max = int(max(dst[:, 0, 1]))
                
                centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                bbox = np.array([x_min, y_min, x_max, y_max])
                return True, bbox, centroid, float(x_max - x_min)
        
        return False, None, None, None
    
    def draw_detection(self, frame: np.ndarray, bbox: Optional[np.ndarray],
                       centroid: Optional[Tuple[int, int]],
                       distance_cm: Optional[float] = None,
                       angle_deg: Optional[float] = None) -> np.ndarray:
        """
        Draw detection overlay on frame.
        
        Args:
            frame: Input BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            centroid: Object center (cx, cy)
            distance_cm: Estimated distance in cm
            angle_deg: Relative angle in degrees
            
        Returns:
            Annotated BGR image
        """
        display = frame.copy()
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Box color based on lock status
            color = (0, 255, 0) if self.target_locked else (0, 165, 255)
            thickness = 2 if self.target_locked else 1
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(display, (cx, cy), 6, (255, 0, 0), -1)
                cv2.circle(display, (cx, cy), 8, (255, 0, 0), 2)
                
                # Crosshair
                cv2.line(display, (cx - 25, cy), (cx + 25, cy), (255, 0, 0), 1)
                cv2.line(display, (cx, cy - 25), (cx, cy + 25), (255, 0, 0), 1)
            
            # Display measurements
            if distance_cm is not None and distance_cm < 500:
                cv2.putText(display, f"Dist: {distance_cm:.1f} cm",
                           (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 0), 2)
            if angle_deg is not None:
                cv2.putText(display, f"Angle: {angle_deg:.1f} deg",
                           (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 0), 2)
        
        return display
